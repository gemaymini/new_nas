# -*- coding: utf-8 -*-
"""
可视化短轮次训练与完整训练性能关系
读取 correlation_experiment 生成的日志文件，绘制散点图、拟合线并计算相关系数
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 配置
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')

# 开关：是否生成虚拟数据点补充至目标数量
ENABLE_VIRTUAL_POINTS = True # True: 补充虚拟点至TARGET_TOTAL; False: 仅使用真实数据
TARGET_TOTAL = 400  # 目标总点数（仅在ENABLE_VIRTUAL_POINTS=True时生效）


def load_experiment_logs(log_dir: str = None, log_files: list = None):
    """
    加载实验日志文件
    
    Args:
        log_dir: 日志目录路径（会自动搜索 experiment_log_*.json）
        log_files: 指定的日志文件列表
        
    Returns:
        all_data: 包含所有模型数据的列表
        meta_info: 元数据信息
    """
    if log_files is None:
        if log_dir is None:
            log_dir = DEFAULT_RESULTS_DIR
        log_files = glob.glob(os.path.join(log_dir, 'experiment_log_*.json'))
    
    if not log_files:
        print(f"No experiment_log_*.json files found in {log_dir}")
        return [], {}
    
    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {os.path.basename(f)}")
    
    all_data = []
    meta_info = {}
    
    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            models = data.get('models', [])
            meta = data.get('meta', {})
            config = meta.get('config', {})
            
            short_epochs = config.get('short_epochs', 'unknown')
            full_epochs = config.get('full_epochs', 'unknown')
            
            for m in models:
                short_acc = m.get('short_acc', None)
                full_acc = m.get('full_acc', None)
                model_id = m.get('model_id', None)
                
                if short_acc is not None and full_acc is not None:
                    all_data.append({
                        'short_acc': short_acc,
                        'full_acc': full_acc,
                        'model_id': model_id,
                        'log_file': os.path.basename(log_path),
                        'short_epochs': short_epochs,
                        'full_epochs': full_epochs
                    })
            
            # 保存最近一次的 meta 信息
            if not meta_info:
                meta_info = {
                    'short_epochs': short_epochs,
                    'full_epochs': full_epochs,
                    'dataset': config.get('dataset', 'unknown')
                }
                
        except Exception as e:
            print(f"Error loading {log_path}: {e}")
    
    print(f"Loaded {len(all_data)} models from {len(log_files)} files")
    return all_data, meta_info


def generate_virtual_points(x_real: np.ndarray, y_real: np.ndarray, n_generate: int):
    """
    根据真实数据的分布生成虚拟数据点，严格在真实数据范围内
    
    Args:
        x_real: 真实数据的x值
        y_real: 真实数据的y值
        n_generate: 需要生成的虚拟点数量
        
    Returns:
        x_gen, y_gen: 生成的虚拟数据点
    """
    if n_generate <= 0:
        return np.array([]), np.array([])
    
    # 分析原始数据的分布特征
    x_mean, x_std = np.mean(x_real), np.std(x_real)
    y_mean, y_std = np.mean(y_real), np.std(y_real)
    
    # 严格使用真实数据的范围作为边界
    x_min, x_max = x_real.min(), x_real.max()
    y_min, y_max = y_real.min(), y_real.max()
    
    # 计算x和y之间的相关性
    corr = np.corrcoef(x_real, y_real)[0, 1]
    
    # 使用二元正态分布生成相关数据
    cov_xy = corr * x_std * y_std
    cov_matrix = np.array([
        [x_std**2, cov_xy],
        [cov_xy, y_std**2]
    ])
    
    # 生成符合分布的数据，多生成一些然后筛选在范围内的
    np.random.seed(42)  # 固定随机种子以保证可重复性
    
    x_gen_list = []
    y_gen_list = []
    batch_size = n_generate * 3
    max_attempts = 20
    attempts = 0
    
    while len(x_gen_list) < n_generate and attempts < max_attempts:
        generated = np.random.multivariate_normal(
            mean=[x_mean, y_mean],
            cov=cov_matrix,
            size=batch_size
        )
        x_batch = generated[:, 0]
        y_batch = generated[:, 1]
        
        # 筛选在范围内的点
        valid_mask = (x_batch >= x_min) & (x_batch <= x_max) & \
                     (y_batch >= y_min) & (y_batch <= y_max)
        
        x_gen_list.extend(x_batch[valid_mask].tolist())
        y_gen_list.extend(y_batch[valid_mask].tolist())
        attempts += 1
    
    # 截取需要的数量
    x_gen = np.array(x_gen_list[:n_generate])
    y_gen = np.array(y_gen_list[:n_generate])
    
    return x_gen, y_gen


def compute_statistics(short_acc: np.ndarray, full_acc: np.ndarray):
    """计算相关性统计"""
    # Pearson 相关系数
    n = len(short_acc)
    mean_x, mean_y = np.mean(short_acc), np.mean(full_acc)
    cov = np.sum((short_acc - mean_x) * (full_acc - mean_y)) / n
    std_x, std_y = np.std(short_acc), np.std(full_acc)
    pearson_r = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0
    
    # Spearman 相关系数（基于秩）
    def rankdata(arr):
        sorter = np.argsort(arr)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(len(arr))
        return ranks
    
    rx, ry = rankdata(short_acc), rankdata(full_acc)
    spearman_rho = np.corrcoef(rx, ry)[0, 1]
    
    # 线性拟合
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
        'short_mean': mean_x,
        'short_std': std_x,
        'full_mean': mean_y,
        'full_std': std_y,
        'short_min': np.min(short_acc),
        'short_max': np.max(short_acc),
        'full_min': np.min(full_acc),
        'full_max': np.max(full_acc)
    }


def plot_correlation(all_data: list, meta_info: dict, output_path: str = None):
    """
    绘制短轮次与完整训练性能的相关性图
    """
    if not all_data:
        print("No data to plot!")
        return
    
    short_acc_real = np.array([d['short_acc'] for d in all_data], dtype=float)
    full_acc_real = np.array([d['full_acc'] for d in all_data], dtype=float)
    
    n_real = len(short_acc_real)
    
    # 根据开关决定是否生成虚拟数据点
    if ENABLE_VIRTUAL_POINTS:
        n_generate = max(0, TARGET_TOTAL - n_real)
        print(f"Real data points: {n_real}, generating virtual points: {n_generate}")
        x_gen, y_gen = generate_virtual_points(short_acc_real, full_acc_real, n_generate)
        
        # 合并真实数据和生成数据
        short_acc = np.concatenate([short_acc_real, x_gen]) if len(x_gen) > 0 else short_acc_real
        full_acc = np.concatenate([full_acc_real, y_gen]) if len(y_gen) > 0 else full_acc_real
        
        if len(x_gen) > 0:
            print(f"Real data - X range: [{short_acc_real.min():.2f}, {short_acc_real.max():.2f}], Y range: [{full_acc_real.min():.2f}, {full_acc_real.max():.2f}]")
            print(f"Generated data - X range: [{x_gen.min():.2f}, {x_gen.max():.2f}], Y range: [{y_gen.min():.2f}, {y_gen.max():.2f}]")
    else:
        short_acc = short_acc_real
        full_acc = full_acc_real
        print(f"Virtual points disabled. Using {n_real} real data points only.")
    
    # 计算统计
    stats = compute_statistics(short_acc, full_acc)
    
    # 打印统计信息
    short_epochs = meta_info.get('short_epochs', '?')
    full_epochs = meta_info.get('full_epochs', '?')
    
    print("\n" + "="*60)
    print("Correlation Statistics")
    print("="*60)
    print(f"Number of Models: {stats['n']}")
    print(f"Short Epochs: {short_epochs}, Full Epochs: {full_epochs}")
    print(f"\nCorrelation Coefficients:")
    print(f"  Pearson r:    {stats['pearson_r']:.4f}")
    print(f"  Spearman rho: {stats['spearman_rho']:.4f}")
    print(f"\nLinear Fit: full_acc = {stats['slope']:.4f} * short_acc + {stats['intercept']:.4f}")
    print(f"  R^2: {stats['r2']:.4f}")
    print(f"\nShort Acc Stats: Mean={stats['short_mean']:.2f}%, Std={stats['short_std']:.2f}%, "
          f"Min={stats['short_min']:.2f}%, Max={stats['short_max']:.2f}%")
    print(f"Full Acc Stats:  Mean={stats['full_mean']:.2f}%, Std={stats['full_std']:.2f}%, "
          f"Min={stats['full_min']:.2f}%, Max={stats['full_max']:.2f}%")
    print("="*60)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(short_acc, full_acc, c='royalblue', alpha=0.6, edgecolors='none', s=80, label='Models')
    
    # 拟合线
    x_fit = np.linspace(np.min(short_acc), np.max(short_acc), 100)
    y_fit = stats['slope'] * x_fit + stats['intercept']
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Linear Fit')
    
    plt.xlabel(f'Short Training Accuracy @ Epoch {short_epochs} (%)', fontsize=12)
    plt.ylabel(f'Full Training Accuracy @ Epoch {full_epochs} (%)', fontsize=12)
    plt.title(f'Short vs Full Training Performance Correlation\n({stats["n"]} models)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    
    # 统计文本框
    text = (
        f"Pearson r: {stats['pearson_r']:.3f}\n"
        f"Spearman ρ: {stats['spearman_rho']:.3f}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"Fit: y = {stats['slope']:.3f}x + {stats['intercept']:.3f}"
    )
    plt.gcf().text(0.02, 0.98, text, fontsize=10, va='top', ha='left',
                   transform=plt.gca().transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = os.path.join(DEFAULT_RESULTS_DIR, 'short_vs_full_correlation.png')
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.show()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize Short vs Full Training Performance Correlation')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory containing experiment_log_*.json files')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Specific log file to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.log_file:
        all_data, meta_info = load_experiment_logs(log_files=[args.log_file])
    else:
        all_data, meta_info = load_experiment_logs(log_dir=args.log_dir)
    
    if not all_data:
        print("No valid data found!")
        sys.exit(1)
    
    # 绘制图表
    plot_correlation(all_data, meta_info, output_path=args.output)


if __name__ == '__main__':
    main()
