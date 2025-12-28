# -*- coding: utf-8 -*-
"""
可视化脚本：生成三种算法对比图

可以使用：
1. 已有的实验结果JSON文件
2. 手动输入的数据
3. 模拟数据（用于演示）

生成类似论文图3-13的精度-参数量对比图
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial import ConvexHull
from datetime import datetime
from typing import List, Tuple, Dict

# 使用英文字体，避免中文字体问题
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = True


def generate_simulated_data():
    """
    生成模拟数据用于演示
    基于图片中的数据分布特征
    """
    np.random.seed(42)
    
    # 三阶段EA: 低参数量 + 高精度 (帕累托最优)
    ts_params = np.random.uniform(0.8, 2.5, 15)
    ts_accs = 95.8 + np.random.uniform(0, 0.8, 15) - (ts_params - 1.5) * 0.05
    ts_accs = np.clip(ts_accs, 95.8, 96.6)
    
    # 传统EA: 中等参数量 + 中等精度
    te_params = np.random.uniform(2.5, 5.5, 15)
    te_accs = 95.8 + np.random.uniform(0, 0.7, 15)
    te_accs = np.clip(te_accs, 95.8, 96.5)
    
    # 随机搜索: 高参数量 + 分散精度
    rs_params = np.random.uniform(4.0, 7.0, 20)
    rs_accs = 95.3 + np.random.uniform(0, 1.0, 20)
    rs_accs = np.clip(rs_accs, 95.3, 96.3)
    
    return {
        'three_stage_ea': {'params': ts_params.tolist(), 'accs': ts_accs.tolist()},
        'traditional_ea': {'params': te_params.tolist(), 'accs': te_accs.tolist()},
        'random_search': {'params': rs_params.tolist(), 'accs': rs_accs.tolist()}
    }


def load_experiment_results(json_path: str) -> dict:
    """从JSON文件加载实验结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {}
    for key in ['three_stage_ea', 'traditional_ea', 'random_search']:
        if key in data:
            models = data[key]
            params = [m['param_count'] for m in models if m['param_count'] > 0]
            accs = [m['accuracy'] for m in models if m['accuracy'] > 0]
            result[key] = {'params': params, 'accs': accs}
        else:
            result[key] = {'params': [], 'accs': []}
    
    return result


def plot_algorithm_comparison(data: dict, 
                              output_path: str = None,
                              show_plot: bool = True,
                              title: str = None,
                              style: str = 'paper'):
    """
    绘制三种算法的对比图
    
    Args:
        data: 包含三种算法数据的字典
        output_path: 输出路径
        show_plot: 是否显示图表
        title: 图表标题
        style: 绘图风格 ('paper' 或 'simple')
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 颜色配置（与论文图片一致）
    colors = {
        'three_stage_ea': '#FF6B6B',   # 红色
        'traditional_ea': '#90EE90',    # 浅绿色
        'random_search': '#6B8EFF'      # 蓝色
    }
    
    labels = {
        'three_stage_ea': 'Three-Stage EA',
        'traditional_ea': 'Traditional EA',
        'random_search': 'Random Search'
    }
    
    markers = {
        'three_stage_ea': 'o',
        'traditional_ea': 'o',
        'random_search': 'o'
    }
    
    def plot_with_hull(params, accs, color, label, marker='o', zorder=1):
        """绘制散点图和凸包"""
        params = np.array(params)
        accs = np.array(accs)
        
        if len(params) == 0:
            return
        
        # 绘制散点
        ax.scatter(params, accs, c=color, label=label, s=100, 
                  alpha=0.9, edgecolors='white', linewidths=1.5, 
                  marker=marker, zorder=zorder+1)
        
        # 绘制凸包
        if len(params) >= 3:
            try:
                points = np.column_stack([params, accs])
                hull = ConvexHull(points)
                
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合
                
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                       color=color, alpha=0.25, zorder=zorder)
                ax.plot(hull_points[:, 0], hull_points[:, 1], 
                       color=color, linewidth=2, alpha=0.6, zorder=zorder)
            except Exception:
                pass
    
    # 按顺序绘制（随机搜索在底层，三阶段EA在顶层）
    plot_order = ['random_search', 'traditional_ea', 'three_stage_ea']
    
    for i, key in enumerate(plot_order):
        if key in data and len(data[key]['params']) > 0:
            plot_with_hull(
                data[key]['params'],
                data[key]['accs'],
                colors[key],
                labels[key],
                markers[key],
                zorder=i
            )
    
    # 设置图表样式
    ax.set_xlabel('Parameters (M)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    
    if title is None:
        title = 'Comparison of Three Search Algorithms on Accuracy vs Parameters'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置坐标轴范围（基于数据自动调整，留出边距）
    all_params = []
    all_accs = []
    for key in data:
        all_params.extend(data[key]['params'])
        all_accs.extend(data[key]['accs'])
    
    if all_params and all_accs:
        param_margin = (max(all_params) - min(all_params)) * 0.1
        acc_margin = (max(all_accs) - min(all_accs)) * 0.15
        
        ax.set_xlim(max(0, min(all_params) - param_margin), 
                   max(all_params) + param_margin)
        ax.set_ylim(min(all_accs) - acc_margin, 
                   max(all_accs) + acc_margin)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存到: {output_path}")
        
        # 同时保存PDF
        pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, dpi=200, bbox_inches='tight')
        print(f"PDF已保存到: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    return fig


def print_statistics(data: dict):
    """Print statistics"""
    print("\n" + "=" * 60)
    print("                    Statistics")
    print("=" * 60)
    
    labels = {
        'three_stage_ea': 'Three-Stage EA',
        'traditional_ea': 'Traditional EA',
        'random_search': 'Random Search'
    }
    
    for key, label in labels.items():
        if key in data and len(data[key]['params']) > 0:
            params = np.array(data[key]['params'])
            accs = np.array(data[key]['accs'])
            
            print(f"\n{label}:")
            print(f"  Samples: {len(params)}")
            print(f"  Accuracy Range: {accs.min():.2f}% - {accs.max():.2f}%")
            print(f"  Mean Accuracy: {accs.mean():.2f}% +/- {accs.std():.2f}%")
            print(f"  Params Range: {params.min():.2f}M - {params.max():.2f}M")
            print(f"  Mean Params: {params.mean():.2f}M +/- {params.std():.2f}M")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='生成三种算法对比可视化图')
    
    parser.add_argument('--json_path', type=str, default=None,
                        help='实验结果JSON文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径')
    parser.add_argument('--use_simulated', action='store_true',
                        help='使用模拟数据')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图表')
    parser.add_argument('--title', type=str, default=None,
                        help='图表标题')
    
    args = parser.parse_args()
    
    # 加载或生成数据
    if args.json_path and os.path.exists(args.json_path):
        print(f"从文件加载数据: {args.json_path}")
        data = load_experiment_results(args.json_path)
    elif args.use_simulated or args.json_path is None:
        print("使用模拟数据生成演示图表...")
        data = generate_simulated_data()
    else:
        print(f"错误: 找不到文件 {args.json_path}")
        return
    
    # 打印统计信息
    print_statistics(data)
    
    # 设置输出路径
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(output_dir, 'experiment_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(output_dir, f'algorithm_comparison_{timestamp}.png')
    
    # 绘制图表
    plot_algorithm_comparison(
        data,
        output_path=args.output,
        show_plot=not args.no_show,
        title=args.title
    )


if __name__ == '__main__':
    main()
