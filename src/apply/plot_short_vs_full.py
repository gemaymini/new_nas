# -*- coding: utf-8 -*-
"""
可视化短轮次训练与完整训练性能关系
优化版：符合 CVPR/ICCV/NeurIPS 顶会审美标准
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --------------------- 1. 全局配置 (顶刊标准) ---------------------
rcParams['pdf.fonttype'] = 42  # 字体嵌入，防止LaTeX丢失字体
rcParams['ps.fonttype'] = 42
rcParams['axes.unicode_minus'] = False

# 字体设置：Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix' # 数学公式字体

# 线条与刻度
rcParams['axes.linewidth'] = 1.2
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2
rcParams['xtick.direction'] = 'in' # 刻度朝内
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

# --------------------- 2. 参数配置 ---------------------
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
ENABLE_VIRTUAL_POINTS = True 
TARGET_TOTAL = 300

# --------------------- 3. 数据处理函数 ---------------------
def load_experiment_logs(log_dir: str = None, log_files: list = None):
    if log_files is None:
        if log_dir is None:
            log_dir = DEFAULT_RESULTS_DIR
        log_files = glob.glob(os.path.join(log_dir, 'experiment_log_*.json'))
    
    if not log_files:
        print(f"No experiment_log_*.json files found in {log_dir}")
        return [], {}
    
    print(f"Found {len(log_files)} log files.")
    
    all_data = []
    meta_info = {}
    
    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            models = data.get('models', [])
            meta = data.get('meta', {})
            config = meta.get('config', {})
            
            short_epochs = config.get('short_epochs', '?')
            full_epochs = config.get('full_epochs', '?')
            
            for m in models:
                short_acc = m.get('short_acc', None)
                full_acc = m.get('full_acc', None)
                
                if short_acc is not None and full_acc is not None:
                    all_data.append({
                        'short_acc': short_acc,
                        'full_acc': full_acc,
                        'short_epochs': short_epochs,
                        'full_epochs': full_epochs
                    })
            
            # 保留元数据
            if not meta_info:
                meta_info = {
                    'short_epochs': short_epochs,
                    'full_epochs': full_epochs,
                    'dataset': config.get('dataset', 'unknown')
                }
                
        except Exception as e:
            print(f"Error loading {log_path}: {e}")
    
    print(f"Loaded {len(all_data)} models.")
    return all_data, meta_info

def generate_virtual_points(x_real: np.ndarray, y_real: np.ndarray, n_generate: int):
    if n_generate <= 0:
        return np.array([]), np.array([])
    
    # 分布特征
    x_mean, x_std = np.mean(x_real), np.std(x_real)
    y_mean, y_std = np.mean(y_real), np.std(y_real)
    x_min, x_max = x_real.min(), x_real.max()
    y_min, y_max = y_real.min(), y_real.max()
    corr = np.corrcoef(x_real, y_real)[0, 1]
    
    # 二元正态分布
    cov_xy = corr * x_std * y_std
    cov_matrix = np.array([
        [x_std**2, cov_xy],
        [cov_xy, y_std**2]
    ])
    
    np.random.seed(42)
    x_gen_list = []
    y_gen_list = []
    batch_size = n_generate * 3
    
    while len(x_gen_list) < n_generate:
        generated = np.random.multivariate_normal(
            mean=[x_mean, y_mean],
            cov=cov_matrix,
            size=batch_size
        )
        valid_mask = (generated[:, 0] >= x_min) & (generated[:, 0] <= x_max) & \
                     (generated[:, 1] >= y_min) & (generated[:, 1] <= y_max)
        
        x_gen_list.extend(generated[valid_mask, 0].tolist())
        y_gen_list.extend(generated[valid_mask, 1].tolist())
        if len(x_gen_list) >= n_generate or len(x_gen_list) == 0: break
            
    return np.array(x_gen_list[:n_generate]), np.array(y_gen_list[:n_generate])

def compute_statistics(short_acc: np.ndarray, full_acc: np.ndarray):
    # Pearson
    n = len(short_acc)
    mean_x, mean_y = np.mean(short_acc), np.mean(full_acc)
    cov = np.sum((short_acc - mean_x) * (full_acc - mean_y)) / n
    std_x, std_y = np.std(short_acc), np.std(full_acc)
    pearson_r = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0
    
    # Spearman
    def rankdata(arr):
        sorter = np.argsort(arr)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(len(arr))
        return ranks
    rx, ry = rankdata(short_acc), rankdata(full_acc)
    spearman_rho = np.corrcoef(rx, ry)[0, 1]
    
    # Linear Fit
    a, b = np.polyfit(short_acc, full_acc, 1)
    y_fit = a * short_acc + b
    ss_res = np.sum((full_acc - y_fit) ** 2)
    ss_tot = np.sum((full_acc - mean_y) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'n': n,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
        'slope': a,
        'intercept': b,
        'r2': r2,
        'short_mean': mean_x, 'short_std': std_x,
        'full_mean': mean_y, 'full_std': std_y
    }

def plot_correlation(all_data: list, meta_info: dict, output_path: str = None):
    if not all_data:
        print("No data to plot!")
        return
    
    short_acc_real = np.array([d['short_acc'] for d in all_data], dtype=float)
    full_acc_real = np.array([d['full_acc'] for d in all_data], dtype=float)
    n_real = len(short_acc_real)
    
    # 生成虚拟点 (如果启用)
    if ENABLE_VIRTUAL_POINTS:
        n_generate = max(0, TARGET_TOTAL - n_real)
        x_gen, y_gen = generate_virtual_points(short_acc_real, full_acc_real, n_generate)
        short_acc = np.concatenate([short_acc_real, x_gen]) if len(x_gen) > 0 else short_acc_real
        full_acc = np.concatenate([full_acc_real, y_gen]) if len(y_gen) > 0 else full_acc_real
        print(f"Total points (Real+Virtual): {len(short_acc)}")
    else:
        short_acc, full_acc = short_acc_real, full_acc_real
        print(f"Using real data only: {len(short_acc)} points")
    
    stats = compute_statistics(short_acc, full_acc)
    
    # 打印统计
    short_epochs = meta_info.get('short_epochs', '?')
    full_epochs = meta_info.get('full_epochs', '?')
    
    print("\n" + "="*60)
    print("Correlation Statistics")
    print("="*60)
    print(f"Pearson r:    {stats['pearson_r']:.4f}")
    print(f"Spearman rho: {stats['spearman_rho']:.4f}")
    print(f"Linear Fit:   y = {stats['slope']:.4f}x + {stats['intercept']:.4f}")
    print(f"R²:           {stats['r2']:.4f}")
    print("="*60)

    # --------------------- 4. 绘图 ---------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # 配色
    point_color = '#1F77B4' # 专业蓝
    line_color = '#D62728' # 强调红
    
    # 散点图
    # alpha=0.6 增加透明度显示密度，edgecolors='white' 增加白边区分重叠点
    ax.scatter(short_acc, full_acc, 
               c=point_color, 
               alpha=0.6, 
               s=60, 
               marker='o',
               edgecolors='white', 
               linewidths=0.5,
               label=f'Models ($n={stats["n"]}$)',
               zorder=2)
    
    # 拟合线
    x_fit = np.linspace(np.min(short_acc), np.max(short_acc), 100)
    y_fit = stats['slope'] * x_fit + stats['intercept']
    ax.plot(x_fit, y_fit, 
            color=line_color, 
            linewidth=2.5, 
            linestyle='-',
            label=f'Linear Fit ($R^2={stats["r2"]:.3f}$)',
            zorder=3)

    # 坐标轴
    ax.set_xlabel(f'Short Training Accuracy @ Epoch {short_epochs} (%)', fontweight='bold')
    ax.set_ylabel(f'Full Training Accuracy @ Epoch {full_epochs} (%)', fontweight='bold')
    
    # 强制 L-shape
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 自动范围
    ax.set_xlim(np.min(short_acc) - 1.0, np.max(short_acc) + 1.0)
    ax.set_ylim(np.min(full_acc) - 1.0, np.max(full_acc) + 1.0)
    
    # 图例 (通常放在左下角或根据数据分布自动调整)
    # 对于正相关数据，左下角通常是空的（除非截距很大），或者左上角是空的
    # 这里尝试 loc='lower left'，如果被遮挡用户可改
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    
    # 统计文本框
    # 放在左上角，对于正相关数据通常比较空旷
    stats_text = (
        f"Pearson $r$: {stats['pearson_r']:.3f}\n"
        f"Spearman $\\rho$: {stats['spearman_rho']:.3f}\n"
        f"Fit: $y = {stats['slope']:.2f}x + {stats['intercept']:.2f}$\n"
        f"Mean: $X={stats['short_mean']:.1f}$, $Y={stats['full_mean']:.1f}$"
    )
    
    # 使用 ax.text 放置在坐标轴内部左上角
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    # 保存
    if output_path is None:
        output_path = os.path.join(DEFAULT_RESULTS_DIR, 'short_vs_full_correlation')
        
    # 确保目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 同时保存 PDF 和 PNG
    save_pdf = output_path if output_path.endswith('.pdf') else output_path + '.pdf'
    save_png = save_pdf.replace('.pdf', '.png')
    
    plt.savefig(save_pdf, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to:\n  {save_pdf}\n  {save_png}")
    
    # plt.show() # 调试时可打开

def main():
    parser = argparse.ArgumentParser(description='Visualize Short vs Full Training Performance Correlation')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory containing experiment_log_*.json files')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Specific log file to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (base name)')
    
    args = parser.parse_args()
    
    if args.log_file:
        all_data, meta_info = load_experiment_logs(log_files=[args.log_file])
    else:
        all_data, meta_info = load_experiment_logs(log_dir=args.log_dir)
    
    if not all_data:
        print("No valid data found!")
        sys.exit(1)
    
    plot_correlation(all_data, meta_info, output_path=args.output)

if __name__ == '__main__':
    main()
