# -*- coding: utf-8 -*-
"""
重训练模型工具
读取pth模型文件，仅使用模型结构（不使用权重），重新训练300轮共10次，计算平均性能
"""
import torch
import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from core.encoding import Individual, Encoder
from models.network import NetworkBuilder
from data.dataset import DatasetLoader
from engine.trainer import NetworkTrainer
from configuration.config import config
from utils.logger import logger


def load_model_encoding(model_path: str) -> list:
    """
    从pth文件加载模型编码
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        encoding: 模型编码列表
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    encoding = None
    
    if isinstance(checkpoint, dict):
        if 'encoding' in checkpoint:
            encoding = checkpoint['encoding']
            print(f"Found encoding in checkpoint")
            
            # 打印元数据
            for k, v in checkpoint.items():
                if k not in ['state_dict', 'encoding', 'history']:
                    print(f"  {k}: {v}")
        else:
            raise ValueError("Checkpoint does not contain 'encoding' key. Cannot extract model structure.")
    else:
        raise ValueError("Unknown checkpoint format. Expected a dict with 'encoding' key.")
    
    return encoding


def build_fresh_network(encoding: list, input_channels: int = 3, num_classes: int = 10):
    """
    从编码构建全新的网络（随机初始化权重）
    
    Args:
        encoding: 模型编码
        input_channels: 输入通道数
        num_classes: 分类数
        
    Returns:
        network: 新构建的网络
    """
    network = NetworkBuilder.build_from_encoding(encoding, input_channels, num_classes)
    return network


def train_once(encoding: list, trainloader, testloader, epochs: int, 
               run_id: int, device: str = None) -> dict:
    """
    进行一次完整训练
    
    Args:
        encoding: 模型编码
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        epochs: 训练轮数
        run_id: 当前运行编号
        device: 设备
        
    Returns:
        results: 训练结果字典
    """
    print(f"\n{'='*60}")
    print(f"Training Run {run_id}")
    print(f"{'='*60}")
    
    # 构建全新网络（随机初始化）
    network = build_fresh_network(encoding)
    param_count = network.get_param_count()
    print(f"Network parameters: {param_count:,}")
    
    # 创建训练器
    trainer = NetworkTrainer(device=device)
    
    # 训练网络
    best_acc, history = trainer.train_network(
        model=network,
        trainloader=trainloader,
        testloader=testloader,
        epochs=epochs
    )
    
    # 最终评估
    final_train_acc = history[-1]['train_acc']
    final_test_acc = history[-1]['test_acc']
    
    print(f"\nRun {run_id} completed:")
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
    
    return {
        'run_id': run_id,
        'best_acc': best_acc,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'param_count': param_count,
        'history': history
    }


def retrain_model(model_path: str, epochs: int = 300, num_runs: int = 10,
                  dataset: str = 'cifar10', device: str = None,
                  save_results: bool = True) -> dict:
    """
    重训练模型多次并计算平均性能
    
    Args:
        model_path: 模型文件路径
        epochs: 每次训练的轮数
        num_runs: 训练次数
        dataset: 数据集名称
        device: 设备
        save_results: 是否保存结果
        
    Returns:
        summary: 汇总结果
    """
    # 加载模型编码
    encoding = load_model_encoding(model_path)
    
    print("\n" + "="*60)
    print("Architecture Encoding:")
    print("="*60)
    print(f"Encoding: {encoding}")
    Encoder.print_architecture(encoding)
    
    # 加载数据集
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    if dataset == 'cifar10':
        trainloader, testloader = DatasetLoader.get_cifar10()
    elif dataset == 'cifar100':
        trainloader, testloader = DatasetLoader.get_cifar100()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"Dataset: {dataset}")
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    # 设置设备
    if device is None:
        device = config.DEVICE
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA not available, using CPU")
    print(f"Device: {device}")
    
    # 进行多次训练
    all_results = []
    best_accs = []
    final_test_accs = []
    
    start_time = datetime.now()
    
    for run in range(1, num_runs + 1):
        result = train_once(
            encoding=encoding,
            trainloader=trainloader,
            testloader=testloader,
            epochs=epochs,
            run_id=run,
            device=device
        )
        all_results.append(result)
        best_accs.append(result['best_acc'])
        final_test_accs.append(result['final_test_acc'])
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 计算统计信息
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    best_acc_mean = np.mean(best_accs)
    best_acc_std = np.std(best_accs)
    final_acc_mean = np.mean(final_test_accs)
    final_acc_std = np.std(final_test_accs)
    
    print(f"\nResults over {num_runs} runs:")
    print(f"  Best Test Accuracy:  {best_acc_mean:.2f}% ± {best_acc_std:.2f}%")
    print(f"  Final Test Accuracy: {final_acc_mean:.2f}% ± {final_acc_std:.2f}%")
    print(f"\nIndividual Best Accuracies: {[f'{acc:.2f}%' for acc in best_accs]}")
    print(f"Individual Final Accuracies: {[f'{acc:.2f}%' for acc in final_test_accs]}")
    print(f"\nMin Best Acc: {min(best_accs):.2f}%, Max Best Acc: {max(best_accs):.2f}%")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Average time per run: {total_time/num_runs/60:.2f} minutes")
    
    # 汇总结果
    summary = {
        'model_path': model_path,
        'encoding': encoding,
        'dataset': dataset,
        'epochs': epochs,
        'num_runs': num_runs,
        'param_count': all_results[0]['param_count'],
        'best_acc_mean': best_acc_mean,
        'best_acc_std': best_acc_std,
        'final_acc_mean': final_acc_mean,
        'final_acc_std': final_acc_std,
        'best_accs': best_accs,
        'final_test_accs': final_test_accs,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'all_results': [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
    }
    
    # 保存结果
    if save_results:
        result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retrain_results')
        os.makedirs(result_dir, exist_ok=True)
        
        result_filename = f"retrain_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(result_dir, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {result_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Retrain a model structure multiple times and compute average performance')
    parser.add_argument('model_path', type=str, help='Path to the .pth model file')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs per run (default: 300)')
    parser.add_argument('--runs', type=int, default=10, help='Number of training runs (default: 10)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], 
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use for training (default: auto)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    retrain_model(
        model_path=args.model_path,
        epochs=args.epochs,
        num_runs=args.runs,
        dataset=args.dataset,
        device=args.device,
        save_results=not args.no_save
    )


if __name__ == '__main__':
    main()
