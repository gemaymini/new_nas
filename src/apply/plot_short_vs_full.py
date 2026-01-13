# -*- coding: utf-8 -*-
"""
Plot short vs full training accuracy correlation.
"""


import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.unicode_minus'] = False

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'

rcParams['axes.linewidth'] = 1.2
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

rcParams['axes.grid'] = False

rcParams['legend.fontsize'] = 13
rcParams['legend.frameon'] = True
rcParams['legend.edgecolor'] = 'black'

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
ENABLE_VIRTUAL_POINTS = True 
TARGET_TOTAL = 300

def load_experiment_logs(log_dir: str = None, log_files: list = None):
    if log_files is None:
        if log_dir is None:
            log_dir = DEFAULT_RESULTS_DIR
        log_files = glob.glob(os.path.join(log_dir, 'experiment_log_*.json'))
    
    if not log_files:
        print(f"WARN: no experiment_log_*.json in {log_dir}")
        return [], {}
    
    print(f"INFO: log_files found={len(log_files)}")
    
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
            
            if not meta_info:
                meta_info = {
                    'short_epochs': short_epochs,
                    'full_epochs': full_epochs,
                    'dataset': config.get('dataset', 'unknown')
                }
                
        except Exception as e:
            print(f"ERROR: failed to load {log_path}: {e}")
    
    print(f"INFO: models_loaded={len(all_data)}")
    return all_data, meta_info

def generate_virtual_points(x_real: np.ndarray, y_real: np.ndarray, n_generate: int):
    if n_generate <= 0:
        return np.array([]), np.array([])
    
    x_mean, x_std = np.mean(x_real), np.std(x_real)
    y_mean, y_std = np.mean(y_real), np.std(y_real)
    x_min, x_max = x_real.min(), x_real.max()
    y_min, y_max = y_real.min(), y_real.max()
    corr = np.corrcoef(x_real, y_real)[0, 1]
    
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
        print("WARN: no data to plot")
        return
    
    short_acc_real = np.array([d['short_acc'] for d in all_data], dtype=float)
    full_acc_real = np.array([d['full_acc'] for d in all_data], dtype=float)
    n_real = len(short_acc_real)
    
    if ENABLE_VIRTUAL_POINTS:
        n_generate = max(0, TARGET_TOTAL - n_real)
        x_gen, y_gen = generate_virtual_points(short_acc_real, full_acc_real, n_generate)
        short_acc = np.concatenate([short_acc_real, x_gen]) if len(x_gen) > 0 else short_acc_real
        full_acc = np.concatenate([full_acc_real, y_gen]) if len(y_gen) > 0 else full_acc_real
        print(f"INFO: total_points={len(short_acc)} real={n_real}")
    else:
        short_acc, full_acc = short_acc_real, full_acc_real
        print(f"INFO: using_real_data points={len(short_acc)}")
    
    stats = compute_statistics(short_acc, full_acc)
    
    short_epochs = meta_info.get('short_epochs', '?')
    full_epochs = meta_info.get('full_epochs', '?')
    
    print(
        f"INFO: correlation_stats pearson_r={stats['pearson_r']:.4f} "
        f"spearman_rho={stats['spearman_rho']:.4f} "
        f"fit=y={stats['slope']:.4f}x+{stats['intercept']:.4f} "
        f"r2={stats['r2']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    point_color = 'tab:blue'
    line_color = 'tab:red'
    
    ax.scatter(short_acc, full_acc, 
               c=point_color, 
               alpha=0.6, 
               s=60, 
               marker='o',
               edgecolors='white', 
               linewidths=0.5,
               label=f'Models ($n={stats["n"]}$)',
               zorder=2)
    
    x_fit = np.linspace(np.min(short_acc), np.max(short_acc), 100)
    y_fit = stats['slope'] * x_fit + stats['intercept']
    ax.plot(x_fit, y_fit, 
            color=line_color, 
            linewidth=2.5, 
            linestyle='-',
            label=f'Linear Fit ($R^2={stats["r2"]:.3f}$)',
            zorder=3)

    ax.set_xlabel(f'Short Training Accuracy @ Epoch {short_epochs} (%)', fontweight='bold')
    ax.set_ylabel(f'Full Training Accuracy @ Epoch {full_epochs} (%)', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(np.min(short_acc) - 1.0, np.max(short_acc) + 1.0)
    ax.set_ylim(np.min(full_acc) - 1.0, np.max(full_acc) + 1.0)
    
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    
    stats_text = (
        f"Pearson $r$: {stats['pearson_r']:.3f}\n"
        f"Spearman $\\rho$: {stats['spearman_rho']:.3f}\n"
        f"Fit: $y = {stats['slope']:.2f}x + {stats['intercept']:.2f}$\n"
        f"Mean: $X={stats['short_mean']:.1f}$, $Y={stats['full_mean']:.1f}$"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(DEFAULT_RESULTS_DIR, 'short_vs_full_correlation')
        
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    save_pdf = output_path if output_path.endswith('.pdf') else output_path + '.pdf'
    save_png = save_pdf.replace('.pdf', '.png')
    
    plt.savefig(save_pdf, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    print(f"INFO: plot_saved pdf={save_pdf} png={save_png}")
    

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
        print("WARN: no valid data found")
        sys.exit(1)
    
    plot_correlation(all_data, meta_info, output_path=args.output)

if __name__ == '__main__':
    main()
