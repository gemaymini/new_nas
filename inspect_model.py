# -*- coding: utf-8 -*-
"""
查看模型架构工具
"""
import torch
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.encoding import Individual, Encoder
from model.network import NetworkBuilder

def inspect_model(model_path: str):
    """
    加载模型并打印其架构信息
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    encoding = None
    state_dict = None
    metadata = {}

    if isinstance(checkpoint, dict):
        if 'encoding' in checkpoint:
            encoding = checkpoint['encoding']
            state_dict = checkpoint.get('state_dict')
            # 提取其他元数据
            for k, v in checkpoint.items():
                if k not in ['state_dict', 'encoding', 'history']:
                    metadata[k] = v
        elif 'state_dict' in checkpoint:
            print("Warning: Checkpoint only contains state_dict (legacy format).")
            state_dict = checkpoint['state_dict']
        else:
            # 假设整个dict就是state_dict
            state_dict = checkpoint
    else:
        print("Error: Unknown checkpoint format.")
        return

    if encoding:
        print("\n" + "="*50)
        print("Model Metadata:")
        print("="*50)
        for k, v in metadata.items():
            print(f"{k}: {v}")
            
        print("\n" + "="*50)
        print("Architecture Encoding:")
        print("="*50)
        print(encoding)
        
        print("\n" + "="*50)
        print("Architecture Details:")
        print("="*50)
        Encoder.print_architecture(encoding)
        
        # 尝试构建网络以打印PyTorch结构
        try:
            print("\n" + "="*50)
            print("PyTorch Network Structure:")
            print("="*50)
            ind = Individual(encoding)
            
            # 尝试推断类别数
            num_classes = 10
            if state_dict:
                if 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].shape[0]
                elif 'classifier.weight' in state_dict:
                    num_classes = state_dict['classifier.weight'].shape[0]
            
            network = NetworkBuilder.build_from_individual(
                ind, input_channels=3, num_classes=num_classes
            )
            print(network)
            
            param_count = network.get_param_count()
            print(f"\nTotal Parameters: {param_count:,}")
            
        except Exception as e:
            print(f"Could not build network for visualization: {e}")

    else:
        print("\nError: No encoding found in checkpoint.")
        print("This appears to be a legacy model file or a raw state_dict.")
        print("Cannot reconstruct architecture structure without encoding.")
        
        if state_dict:
            print("\nState Dict Keys (Layer Names):")
            for k in list(state_dict.keys())[:20]:
                print(k)
            if len(state_dict) > 20:
                print("...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect NAS Model Architecture')
    parser.add_argument('model_path', type=str, help='Path to model .pth file')
    
    args = parser.parse_args()
    inspect_model(args.model_path)
