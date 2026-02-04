# -*- coding: utf-8 -*-
"""
根据 Cell 编码评估模型的 NTK 条件数

本脚本用于根据给定的 Normal Cell 和 Reduction Cell 编码构建网络并计算 NTK 条件数。
NTK 条件数是神经网络可训练性的重要指标，数值越小表示网络越容易训练。

使用方法:
1. 从 JSON 文件读取:
   python evaluate_ntk.py --encoding_file path/to/encoding.json

2. 直接指定编码列表:
   python evaluate_ntk.py --normal_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3" --reduction_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3"

3. 自定义评估参数:
   python evaluate_ntk.py --encoding_file encoding.json --num_runs 3 --batch_size 64

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

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration.config import config
from core.encoding import CellEncoding, Individual
from models.network import DARTSNetwork, NetworkBuilder
from engine.evaluator import NTKEvaluator, clear_gpu_memory
from utils.logger import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='根据 Cell 编码评估 DARTS 网络的 NTK 条件数',
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
    
    # NTK 评估参数
    parser.add_argument(
        '--num_runs', type=int, default=3,
        help='NTK 评估运行次数，取平均值 (默认: 3)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help=f'NTK 评估的批次大小 (默认: {config.NTK_BATCH_SIZE})'
    )
    parser.add_argument(
        '--num_batch', type=int, default=1,
        help='用于计算 NTK 的 batch 数量 (默认: 1)'
    )
    parser.add_argument(
        '--recalbn', type=int, default=0,
        help='重新计算 BatchNorm 统计量的 batch 数 (默认: 0, 不重新计算)'
    )
    
    # 网络配置
    parser.add_argument(
        '--cells_per_stage', type=int, default=None,
        help=f'每个 Stage 的 Cell 数量 (默认: {config.NTK_CELLS_PER_STAGE})'
    )
    parser.add_argument(
        '--init_channels', type=int, default=None,
        help=f'初始通道数 (默认: {config.INIT_CHANNELS})'
    )
    
    # 设备选择
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['cuda', 'cpu'],
        help=f'计算设备 (默认: {config.DEVICE})'
    )
    
    # 输出选项
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='显示详细输出'
    )
    parser.add_argument(
        '--output_file', type=str, default=None,
        help='将结果保存到 JSON 文件'
    )
    
    return parser.parse_args()


def load_encoding_from_file(filepath: str) -> dict:
    """从 JSON 文件加载编码"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'normal_cell' not in data or 'reduction_cell' not in data:
        raise ValueError("JSON 文件必须包含 'normal_cell' 和 'reduction_cell' 字段")
    
    return {
        'normal_cell': data['normal_cell'],
        'reduction_cell': data['reduction_cell']
    }


def parse_encoding_string(encoding_str: str) -> list:
    """解析逗号分隔的编码字符串"""
    return [int(x.strip()) for x in encoding_str.split(',')]


def evaluate_ntk(encoding: dict, args) -> dict:
    """
    评估模型的 NTK 条件数
    
    Args:
        encoding: {'normal_cell': [...], 'reduction_cell': [...]}
        args: 命令行参数
    
    Returns:
        包含评估结果的字典
    """
    # 构建 Cell 编码
    normal_cell = CellEncoding.from_list(encoding['normal_cell'])
    reduction_cell = CellEncoding.from_list(encoding['reduction_cell'])
    
    # 验证编码
    if not normal_cell.validate():
        raise ValueError("Normal Cell 编码无效")
    if not reduction_cell.validate():
        raise ValueError("Reduction Cell 编码无效")
    
    # 创建 Individual
    individual = Individual(normal_cell, reduction_cell)
    
    # 配置网络参数
    cells_per_stage = args.cells_per_stage or config.NTK_CELLS_PER_STAGE
    init_channels = args.init_channels or config.INIT_CHANNELS
    
    # 构建网络
    print("\n" + "=" * 60)
    print("构建网络...")
    network = NetworkBuilder.build_from_individual(
        individual,
        num_classes=config.NTK_NUM_CLASSES,
        cells_per_stage=cells_per_stage,
        init_channels=init_channels,
        enable_dropout=False  # NTK 评估时禁用 Dropout
    )
    
    param_count = network.get_param_count()
    print(f"网络参数量: {param_count:,}")
    print(f"Cells per Stage: {cells_per_stage}")
    print(f"Init Channels: {init_channels}")
    
    # 打印网络结构信息
    if args.verbose:
        print("\n--- Normal Cell ---")
        print(normal_cell)
        print("\n--- Reduction Cell ---")
        print(reduction_cell)
    
    # 创建 NTK 评估器
    device = args.device or config.DEVICE
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'
    
    batch_size = args.batch_size or config.NTK_BATCH_SIZE
    
    evaluator = NTKEvaluator(
        input_size=config.NTK_INPUT_SIZE,
        num_classes=config.NTK_NUM_CLASSES,
        batch_size=batch_size,
        device=device,
        recalbn=args.recalbn,
        num_batch=args.num_batch
    )
    
    # 执行 NTK 评估
    print(f"\n计算 NTK 条件数 (运行 {args.num_runs} 次取平均)...")
    print(f"设备: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Batch: {args.num_batch}")
    
    start_time = time.time()
    
    # 计算 NTK 分数
    ntk_score = evaluator.compute_ntk_score(
        network, 
        param_count=param_count,
        num_runs=args.num_runs
    )
    
    elapsed_time = time.time() - start_time
    
    # 清理 GPU 内存
    del network
    clear_gpu_memory()
    
    # 整理结果
    result = {
        'encoding': encoding,
        'ntk_condition_number': ntk_score,
        'param_count': param_count,
        'evaluation_time_seconds': round(elapsed_time, 2),
        'config': {
            'num_runs': args.num_runs,
            'batch_size': batch_size,
            'num_batch': args.num_batch,
            'recalbn': args.recalbn,
            'cells_per_stage': cells_per_stage,
            'init_channels': init_channels,
            'device': device
        }
    }
    
    return result


def print_results(result: dict):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("NTK 评估结果")
    print("=" * 60)
    
    ntk_score = result['ntk_condition_number']
    param_count = result['param_count']
    eval_time = result['evaluation_time_seconds']
    
    print(f"\nNTK 条件数: {ntk_score:.4f}")
    print(f"参数量: {param_count:,}")
    print(f"评估耗时: {eval_time:.2f} 秒")
    
    # NTK 分数解读
    print("\n--- NTK 条件数解读 ---")
    if ntk_score < 100:
        print("优秀: 网络具有良好的可训练性")
    elif ntk_score < 1000:
        print("良好: 网络可训练性较好")
    elif ntk_score < 10000:
        print("一般: 网络可能需要仔细调参")
    elif ntk_score < 100000:
        print("较差: 网络可训练性可能存在问题")
    else:
        print("警告: 网络可能无法有效训练 (死网络或梯度问题)")
    
    print("=" * 60)


def save_results(result: dict, output_file: str):
    """保存结果到 JSON 文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载编码
    if args.encoding_file:
        print(f"从文件加载编码: {args.encoding_file}")
        encoding = load_encoding_from_file(args.encoding_file)
    else:
        if not args.reduction_cell:
            print("错误: 使用 --normal_cell 时必须同时指定 --reduction_cell")
            sys.exit(1)
        
        encoding = {
            'normal_cell': parse_encoding_string(args.normal_cell),
            'reduction_cell': parse_encoding_string(args.reduction_cell)
        }
    
    print(f"Normal Cell 编码: {encoding['normal_cell']}")
    print(f"Reduction Cell 编码: {encoding['reduction_cell']}")
    
    # 执行 NTK 评估
    try:
        result = evaluate_ntk(encoding, args)
        
        # 打印结果
        print_results(result)
        
        # 保存结果
        if args.output_file:
            save_results(result, args.output_file)
        
        return result
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
