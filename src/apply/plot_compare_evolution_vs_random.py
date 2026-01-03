# -*- coding: utf-8 -*-
"""
可视化脚本：Aging Evolution vs Random Search 对比
优化版：符合 CVPR/ICCV/NeurIPS 顶会审美标准
修改：第一张图的图例放置于右上角
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --------------------- 1. 全局配置 (顶刊标准) ---------------------
rcParams['pdf.fonttype'] = 42  # 字体嵌入
rcParams['ps.fonttype'] = 42
rcParams['axes.unicode_minus'] = False

# 字体设置：Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'

# 线条与刻度
rcParams['axes.linewidth'] = 1.2
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# 关键：去除上边框和右边框 (L-shape)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# 去除网格
rcParams['axes.grid'] = False

# 图例
rcParams['legend.fontsize'] = 13
rcParams['legend.frameon'] = True
rcParams['legend.edgecolor'] = 'black'

# --------------------- 2. 颜色定义 ---------------------
COLOR_PROPOSED = '#D62728'  # Red (Aging Evolution - SOTA)
COLOR_BASELINE = '#7F7F7F'  # Grey (Random Search - Baseline)

LABEL_PROPOSED = 'Aging Evolution (Ours)'
LABEL_BASELINE = 'Random Search'

def load_data(json_path: str):
    """加载实验 JSON 数据"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return (
        data.get('evolution_curve', []),
        data.get('random_curve', []),
        data.get('evolution_all_ntk', []),
        data.get('random_all_ntk', [])
    )

def plot_comparison(evolution_curve, random_curve, 
                    evolution_all_ntk, random_all_ntk,
                    output_path):
    """
    绘制对比图
    """
    # 过滤无效值 (假设 100000 为无效/大值)
    def filter_values(arr):
        return [v for v in arr if v < 100000]

    evo_curve_filt = filter_values(evolution_curve)
    rand_curve_filt = filter_values(random_curve)
    evo_all_filt = filter_values(evolution_all_ntk)
    rand_all_filt = filter_values(random_all_ntk)

    if not evo_curve_filt or not rand_curve_filt:
        print("Error: Not enough valid data points found.")
        return

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.35) 

    # --------------------- Subplot 1: Convergence Curve (对数坐标) ---------------------
    steps_evo = np.arange(1, len(evolution_curve) + 1)
    steps_rand = np.arange(1, len(random_curve) + 1)

    # 绘制曲线
    ax1.semilogy(steps_evo, evolution_curve, color=COLOR_PROPOSED, linewidth=2.5, label=LABEL_PROPOSED)
    ax1.semilogy(steps_rand, random_curve, color=COLOR_BASELINE, linewidth=2.0, linestyle='--', label=LABEL_BASELINE)

    # 标注最终值
    final_evo = evolution_curve[-1]
    final_rand = random_curve[-1]
    
    # 略微偏移文本，防止与图例重叠
    ax1.text(len(steps_evo), final_evo, f' {final_evo:.1f}', 
             verticalalignment='center', color=COLOR_PROPOSED, fontweight='bold')
    ax1.text(len(steps_rand), final_rand, f' {final_rand:.1f}', 
             verticalalignment='center', color=COLOR_BASELINE)

    # 样式设置
    ax1.set_xlabel('Evaluation Count', fontweight='bold')
    ax1.set_ylabel('Best NTK Condition Number', fontweight='bold')
    ax1.set_title('(a) Convergence Curve', fontsize=14, loc='left', pad=10)
    
    # [修改点] 将图例放在右上角
    ax1.legend(loc='upper right', frameon=True, fancybox=False)
    
    # 强制 L-shape
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --------------------- Subplot 2: NTK Distribution Histogram (对数 bins) ---------------------
    # 使用对数分bin，更好展示跨数量级的分布
    all_valid = evo_all_filt + rand_all_filt
    if all_valid:
        log_min = np.log10(max(min(all_valid), 1))
        log_max = np.log10(max(all_valid))
        bins = np.logspace(log_min, log_max, 25) # 25个bins
        
        ax2.hist(evo_all_filt, bins=bins, alpha=0.6, color=COLOR_PROPOSED, 
                 label=LABEL_PROPOSED, edgecolor='white', linewidth=0.5)
        ax2.hist(rand_all_filt, bins=bins, alpha=0.6, color=COLOR_BASELINE, 
                 label=LABEL_BASELINE, edgecolor='white', linewidth=0.5)
        
        ax2.set_xscale('log')
    
    ax2.set_xlabel('NTK Condition Number', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('(b) NTK Distribution of All Samples', fontsize=14, loc='left', pad=10)
    ax2.legend(loc='upper right', frameon=True, fancybox=False)
    
    # 强制 L-shape
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 保存
    base_path = output_path.rsplit('.', 1)[0]
    plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_path + '.pdf', bbox_inches='tight')
    print(f"Saved to: {base_path}.pdf")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Aging Evolution vs Random Search')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the JSON log file (e.g., comparison_curves_xxx.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: same dir as json)')
    
    args = parser.parse_args()
    
    # 1. 加载数据
    print(f"Loading data from {args.json_path}...")
    evo_curve, rand_curve, evo_all, rand_all = load_data(args.json_path)
    
    # 2. 计算简单统计
    min_evo = min([v for v in evo_curve if v < 100000]) if any(v < 100000 for v in evo_curve) else 999999
    min_rand = min([v for v in rand_curve if v < 100000]) if any(v < 100000 for v in rand_curve) else 999999
    
    improvement = (min_rand - min_evo) / min_rand * 100 if min_rand > 0 else 0
    print(f"Best NTK (Aging Evolution): {min_evo:.2f}")
    print(f"Best NTK (Random Search):   {min_rand:.2f}")
    print(f"Improvement: {improvement:.2f}%")
    
    # 3. 确定输出路径
    if args.output is None:
        json_dir = os.path.dirname(args.json_path)
        json_name = os.path.splitext(os.path.basename(args.json_path))[0]
        args.output = os.path.join(json_dir, f"{json_name}_vis.png")
    
    # 4. 绘图
    plot_comparison(evo_curve, rand_curve, evo_all, rand_all, args.output)

if __name__ == '__main__':
    main()
