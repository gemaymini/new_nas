# -*- coding: utf-8 -*-
"""
根据 Cell 编码完整训练模型

本脚本用于根据给定的 Normal Cell 和 Reduction Cell 编码构建网络并完整训练。
支持从命令行参数或 JSON 文件读取 Cell 编码。

使用方法:
1. 直接指定编码列表:
   python train_from_encoding.py --normal_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3" --reduction_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3"

2. 从 JSON 文件读取:
   python train_from_encoding.py --encoding_file path/to/encoding.json

3. 自定义训练参数:
   python train_from_encoding.py --normal_cell "..." --reduction_cell "..." --epochs 600 --lr 0.025 --batch_size 128

JSON 文件格式:
{
    "normal_cell": [0, 2, 1, 3, 0, 4, 2, 5, 1, 6, 3, 7, 2, 2, 0, 3],
    "reduction_cell": [0, 2, 1, 3, 0, 4, 2, 5, 1, 6, 3, 7, 2, 2, 0, 3]
}
"""

import os
import sys
import json
import argparse
import torch
import time
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration.config import config
from core.encoding import CellEncoding, Individual
from models.network import DARTSNetwork, NetworkBuilder
from engine.trainer import NetworkTrainer
from data.dataset import DatasetLoader
from utils.logger import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='根据 Cell 编码训练 DARTS 网络',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 编码输入方式
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--encoding_file', type=str,
        help='包含 Cell 编码的 JSON 文件路径'
    )
    input_group.add_argument(
        '--normal_cell', type=str,
        help='Normal Cell 编码 (逗号分隔的整数列表)'
    )
    
    parser.add_argument(
        '--reduction_cell', type=str,
        help='Reduction Cell 编码 (逗号分隔的整数列表)，与 --normal_cell 一起使用'
    )
    
    # 数据集选择
    parser.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['cifar10', 'cifar100'],
        help='训练使用的数据集 (默认: cifar10)'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs', type=int, default=None,
        help=f'训练轮数 (默认: {config.FULL_TRAIN_EPOCHS})'
    )
    parser.add_argument(
        '--lr', '--learning_rate', type=float, default=None,
        help=f'学习率 (默认: {config.LEARNING_RATE})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help=f'批次大小 (默认: {config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--momentum', type=float, default=None,
        help=f'动量 (默认: {config.MOMENTUM})'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=None,
        help=f'权重衰减 (默认: {config.WEIGHT_DECAY})'
    )
    
    # 早停参数
    parser.add_argument(
        '--patience', type=int, default=None,
        help=f'早停耐心值 (默认: {config.EARLY_STOPPING_PATIENCE})'
    )
    parser.add_argument(
        '--min_delta', type=float, default=None,
        help=f'早停最小改进阈值 (默认: {config.EARLY_STOPPING_MIN_DELTA})'
    )
    
    # 网络结构参数
    parser.add_argument(
        '--init_channels', type=int, default=None,
        help=f'初始通道数 (默认: {config.INIT_CHANNELS})'
    )
    parser.add_argument(
        '--cells_per_stage', type=int, default=None,
        help=f'每个 Stage 的 Cell 数量 (默认: {config.CELLS_PER_STAGE})'
    )
    parser.add_argument(
        '--num_stages', type=int, default=None,
        help=f'Stage 数量 (默认: {config.NUM_STAGES})'
    )
    
    # 输出设置
    parser.add_argument(
        '--output_dir', type=str, default='./checkpoints',
        help='模型保存目录 (默认: ./checkpoints)'
    )
    parser.add_argument(
        '--model_name', type=str, default=None,
        help='模型保存名称 (默认: 自动生成)'
    )
    
    # 其他选项
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['cuda', 'cpu'],
        help=f'训练设备 (默认: {config.DEVICE})'
    )
    parser.add_argument(
        '--num_workers', type=int, default=None,
        help=f'数据加载线程数 (默认: {config.NUM_WORKERS})'
    )
    parser.add_argument(
        '--save_history', action='store_true',
        help='保存训练历史记录'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.normal_cell and not args.reduction_cell:
        parser.error("--reduction_cell 必须与 --normal_cell 一起指定")
    
    return args


def parse_cell_encoding(encoding_str: str) -> list:
    """
    解析 Cell 编码字符串
    
    Args:
        encoding_str: 逗号分隔的整数字符串，如 "0,2,1,3,0,4,2,5"
    
    Returns:
        整数列表
    """
    try:
        encoding = [int(x.strip()) for x in encoding_str.split(',')]
        return encoding
    except ValueError as e:
        raise ValueError(f"无法解析编码字符串: {encoding_str}. 错误: {e}")


def load_encoding_from_file(filepath: str) -> tuple:
    """
    从 JSON 文件加载 Cell 编码
    
    Args:
        filepath: JSON 文件路径
    
    Returns:
        (normal_cell_encoding, reduction_cell_encoding) 元组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"编码文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'normal_cell' not in data:
        raise KeyError("JSON 文件中缺少 'normal_cell' 字段")
    if 'reduction_cell' not in data:
        raise KeyError("JSON 文件中缺少 'reduction_cell' 字段")
    
    return data['normal_cell'], data['reduction_cell']


def print_cell_info(cell_encoding: CellEncoding, cell_type: str):
    """打印 Cell 编码信息"""
    print(f"\n{'='*60}")
    print(f"{cell_type} Cell 编码详情:")
    print(f"{'='*60}")
    
    for node_idx, node_edges in enumerate(cell_encoding.edges):
        print(f"  Node {node_idx}:")
        for edge_idx, edge in enumerate(node_edges):
            source_name = f"Input_{edge.source}" if edge.source < 2 else f"Node_{edge.source - 2}"
            op_name = config.OPERATIONS[edge.op_id]
            print(f"    Edge {edge_idx}: {source_name} -> {op_name}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print(" DARTS 网络训练脚本")
    print(" Train DARTS Network from Cell Encoding")
    print("=" * 70)
    
    # 1. 加载或解析编码
    print("\n[1/5] 加载 Cell 编码...")
    
    if args.encoding_file:
        print(f"从文件加载: {args.encoding_file}")
        normal_encoding, reduction_encoding = load_encoding_from_file(args.encoding_file)
    else:
        print("从命令行参数加载")
        normal_encoding = parse_cell_encoding(args.normal_cell)
        reduction_encoding = parse_cell_encoding(args.reduction_cell)
    
    print(f"  Normal Cell 编码长度: {len(normal_encoding)}")
    print(f"  Reduction Cell 编码长度: {len(reduction_encoding)}")
    
    # 2. 创建 CellEncoding 对象
    try:
        normal_cell = CellEncoding.from_list(normal_encoding)
        reduction_cell = CellEncoding.from_list(reduction_encoding)
    except ValueError as e:
        print(f"错误: 编码格式不正确 - {e}")
        print(f"期望的编码长度: {config.NUM_NODES * config.EDGES_PER_NODE * 2}")
        sys.exit(1)
    
    # 验证编码有效性
    if not normal_cell.validate():
        print("错误: Normal Cell 编码验证失败")
        sys.exit(1)
    if not reduction_cell.validate():
        print("错误: Reduction Cell 编码验证失败")
        sys.exit(1)
    
    print("  编码验证通过 ✓")
    
    # 打印 Cell 详情
    print_cell_info(normal_cell, "Normal")
    print_cell_info(reduction_cell, "Reduction")
    
    # 3. 构建网络
    print("\n[2/5] 构建网络...")
    
    # 确定数据集的类别数
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # 构建网络参数
    network_kwargs = {
        'num_classes': num_classes,
        'enable_dropout': True
    }
    if args.init_channels is not None:
        network_kwargs['init_channels'] = args.init_channels
    if args.cells_per_stage is not None:
        network_kwargs['cells_per_stage'] = args.cells_per_stage
    if args.num_stages is not None:
        network_kwargs['num_stages'] = args.num_stages
    
    model = NetworkBuilder.build(normal_cell, reduction_cell, **network_kwargs)
    
    # 打印网络信息
    param_count = model.get_param_count()
    print(f"  模型参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"  Cell 总数: {model.num_cells}")
    print(f"  分类数: {num_classes}")
    
    # 4. 加载数据集
    print("\n[3/5] 加载数据集...")
    
    batch_size = args.batch_size or config.BATCH_SIZE
    num_workers = args.num_workers or config.NUM_WORKERS
    
    if args.dataset == 'cifar10':
        trainloader, testloader = DatasetLoader.get_cifar10(batch_size, num_workers)
        print(f"  数据集: CIFAR-10")
    else:
        trainloader, testloader = DatasetLoader.get_cifar100(batch_size, num_workers)
        print(f"  数据集: CIFAR-100")
    
    print(f"  训练集大小: {len(trainloader.dataset):,}")
    print(f"  测试集大小: {len(testloader.dataset):,}")
    print(f"  批次大小: {batch_size}")
    
    # 5. 训练网络
    print("\n[4/5] 开始训练...")
    
    device = args.device or config.DEVICE
    epochs = args.epochs or config.FULL_TRAIN_EPOCHS
    lr = args.lr or config.LEARNING_RATE
    momentum = args.momentum or config.MOMENTUM
    weight_decay = args.weight_decay or config.WEIGHT_DECAY
    patience = args.patience or config.EARLY_STOPPING_PATIENCE
    min_delta = args.min_delta or config.EARLY_STOPPING_MIN_DELTA
    
    print(f"  设备: {device}")
    print(f"  最大训练轮数: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  动量: {momentum}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  早停耐心值: {patience}")
    print(f"  早停最小改进: {min_delta}")
    
    trainer = NetworkTrainer(device=device)
    
    start_time = time.time()
    best_acc, history = trainer.train_network(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta
    )
    training_time = time.time() - start_time
    
    print(f"\n训练完成!")
    print(f"  最佳测试准确率: {best_acc:.2f}%")
    print(f"  总训练时间: {training_time/3600:.2f} 小时")
    
    # 6. 保存模型和结果
    print("\n[5/5] 保存模型...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成模型名称
    if args.model_name:
        model_name = args.model_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"darts_trained_{args.dataset}_{timestamp}"
    
    # 保存模型权重
    model_path = os.path.join(args.output_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'normal_cell_encoding': normal_encoding,
        'reduction_cell_encoding': reduction_encoding,
        'best_accuracy': best_acc,
        'training_time': training_time,
        'config': {
            'dataset': args.dataset,
            'num_classes': num_classes,
            'epochs': epochs,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'init_channels': args.init_channels or config.INIT_CHANNELS,
            'cells_per_stage': args.cells_per_stage or config.CELLS_PER_STAGE,
            'num_stages': args.num_stages or config.NUM_STAGES,
        }
    }, model_path)
    print(f"  模型已保存: {model_path}")
    
    # 保存训练历史
    if args.save_history:
        history_path = os.path.join(args.output_dir, f"{model_name}_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'history': history,
                'best_accuracy': best_acc,
                'training_time': training_time,
                'normal_cell_encoding': normal_encoding,
                'reduction_cell_encoding': reduction_encoding
            }, f, indent=2, ensure_ascii=False)
        print(f"  训练历史已保存: {history_path}")
    
    # 保存编码信息 (便于复现)
    encoding_path = os.path.join(args.output_dir, f"{model_name}_encoding.json")
    with open(encoding_path, 'w', encoding='utf-8') as f:
        json.dump({
            'normal_cell': normal_encoding,
            'reduction_cell': reduction_encoding,
            'accuracy': best_acc,
            'param_count': param_count
        }, f, indent=2, ensure_ascii=False)
    print(f"  编码信息已保存: {encoding_path}")
    
    print("\n" + "=" * 70)
    print(" 训练完成!")
    print(f" 最佳准确率: {best_acc:.2f}%")
    print(f" 模型路径: {model_path}")
    print("=" * 70)
    
    return best_acc


if __name__ == '__main__':
    main()
