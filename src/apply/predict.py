# -*- coding: utf-8 -*-
"""
模型推理脚本
支持 CIFAR-10 和 CIFAR-100
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from core.encoding import Individual
from models.network import NetworkBuilder

CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
)

CIFAR100_CLASSES = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 
    'worm'
)

import argparse

def predict_image(model_path: str, image_path: str, encoding_str: str = None, device: str = 'cpu'):
    """
    加载模型并对图像进行预测
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    # 1. 加载模型 checkpoint
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 检查checkpoint格式
    encoding = None
    state_dict = None
    
    if isinstance(checkpoint, dict) and 'encoding' in checkpoint:
        encoding = checkpoint['encoding']
        state_dict = checkpoint['state_dict']
        acc = checkpoint.get('accuracy', 'N/A')
        print(f"Model Accuracy on Validation: {acc}")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 兼容只保存了state_dict但没有encoding的情况
        state_dict = checkpoint['state_dict']
        print("Warning: Checkpoint only contains state_dict.")
    elif isinstance(checkpoint, dict) and 'fc.weight' in checkpoint:
        # 兼容直接保存state_dict的情况
        state_dict = checkpoint
        print("Warning: Checkpoint is a raw state_dict.")
    else:
        print("Error: Unknown checkpoint format.")
        return

    if encoding is None:
        if encoding_str:
            try:
                # 尝试解析用户提供的encoding字符串
                import ast
                encoding = ast.literal_eval(encoding_str)
                print(f"Using provided encoding: {encoding}")
            except Exception as e:
                print(f"Error parsing provided encoding: {e}")
                return
        else:
            print("Error: The .pth file does not contain architecture encoding.")
            print("For legacy models, please provide the encoding manually using --encoding argument.")
            print("Example: python predict.py model.pth image.jpg --encoding \"[3, 2, 2, 2, ...]\"")
            return


    # 2. 自动推断类别数量和类别标签
    num_classes = 10 # Default
    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"Detected {num_classes} classes from model weights.")
    elif 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
        print(f"Detected {num_classes} classes from model weights (classifier).")
    
    classes = None
    if num_classes == 10:
        classes = CIFAR10_CLASSES
        print("Using CIFAR-10 class labels.")
    elif num_classes == 100:
        classes = CIFAR100_CLASSES
        print("Using CIFAR-100 class labels.")
    else:
        print(f"Warning: Unknown number of classes {num_classes}. Using numerical labels.")
        classes = [str(i) for i in range(num_classes)]

    # 3. 重建网络架构
    try:
        # 创建临时Individual对象用于构建网络
        ind = Individual(encoding)
        input_channels = 3
        
        network = NetworkBuilder.build_from_individual(
            ind, input_channels=input_channels, num_classes=num_classes
        )
        network.load_state_dict(state_dict)
        network.to(device)
        network.eval()
        print("Network built and weights loaded successfully.")
    except Exception as e:
        print(f"Error building network: {e}")
        return

    # 4. 图像预处理
    try:
        # CIFAR 标准预处理 (Resize to 32x32)
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 5. 推理
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

    print(f"\nPrediction Result:")
    print(f"Image: {image_path}")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # 输出前3个可能的类别
    print("\nTop 3 Predictions:")
    top_k = min(3, num_classes)
    top_prob, top_idx = torch.topk(probabilities, top_k)
    for i in range(top_k):
        idx_val = top_idx[0][i].item()
        if idx_val < len(classes):
            cls = classes[idx_val]
        else:
            cls = f"Class {idx_val}"
        prob = top_prob[0][i].item()
        print(f"{i+1}. {cls}: {prob:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAS Model Prediction')
    parser.add_argument('model_path', type=str, help='Path to model .pth file')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--encoding', type=str, default=None, help='Architecture encoding list (required for legacy models)')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predict_image(args.model_path, args.image_path, args.encoding, device)
