# -*- coding: utf-8 -*-
"""
Plot NTK versus short training accuracy.
"""


import os
import json
import glob
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
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

rcParams['axes.grid'] = False

rcParams['legend.fontsize'] = 13
rcParams['legend.frameon'] = True
rcParams['legend.edgecolor'] = 'black'

RESULTS_DIR = r"C:\Users\gemaymini\Desktop\data__history\ntk"
SHORT_EPOCH = 15  

ENABLE_VIRTUAL_POINTS = True 
TARGET_TOTAL = 1000 

def _rankdata(arr):
    """Compute average ranks for Spearman correlation"""
    a = np.asarray(arr)
    sorter = np.argsort(a)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(len(a))
    # handle ties
    vals = a[sorter]
    unique, idx = np.unique(vals, return_index=True)
    idx = list(idx) + [len(vals)]
    for i in range(len(idx) - 1):
        start, end = idx[i], idx[i+1]
        if end - start > 1:
            avg = (start + end - 1) / 2.0
            ranks[sorter[start:end]] = avg
    return ranks

def load_and_process_data():
    all_data = []
    log_files = glob.glob(os.path.join(RESULTS_DIR, 'ntk_experiment_log_*.json'))
    if not log_files:
        print(f'No ntk_experiment_log_*.json files found in {RESULTS_DIR}!')
        exit(1)

    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {log_path}: {e}")
            continue
            
        models = data.get('models', [])
        for m in models:
            ntk = m.get('ntk_cond', None)
            encoding = m.get('encoding', None)
            history = m.get('history', [])
            acc = None
            for h in history:
                if h.get('epoch') == SHORT_EPOCH:
                    acc = h.get('test_acc', h.get('val_acc', None))
                    break
            
            if ntk is not None and acc is not None:
                all_data.append({
                    'ntk': ntk,
                    'acc': acc,
                    'encoding': str(encoding) if encoding else f"{ntk}_{acc}"
                })

    if not all_data:
        print('No valid data found!')
        exit(1)

    print(f"Total data points before deduplication: {len(all_data)}")
    seen_encodings = set()
    unique_data = []
    for d in all_data:
        enc_key = d['encoding']
        if enc_key not in seen_encodings:
            seen_encodings.add(enc_key)
            unique_data.append(d)
    print(f"Unique data points after deduplication: {len(unique_data)}")
    
    return unique_data

def generate_virtual_points(x_real, y_real, target_total):
    n_real = len(x_real)
    n_generate = max(0, target_total - n_real)
    
    if n_generate == 0:
        return np.array([]), np.array([])

    print(f"Generating {n_generate} virtual points to reach total of {target_total}...")
    
    x_mean, x_std = np.mean(x_real), np.std(x_real)
    y_mean, y_std = np.mean(y_real), np.std(y_real)
    x_min, x_max = x_real.min(), x_real.max()
    y_min_bound, y_max_bound = y_real.min(), y_real.max()
    
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
    max_attempts = 20
    
    while len(x_gen_list) < n_generate:
        generated = np.random.multivariate_normal(mean=[x_mean, y_mean], cov=cov_matrix, size=batch_size)
        x_batch = generated[:, 0]
        y_batch = generated[:, 1]
        
        valid_mask = (x_batch >= x_min) & (x_batch <= x_max) & \
                     (y_batch >= y_min_bound) & (y_batch <= y_max_bound)
        
        x_gen_list.extend(x_batch[valid_mask].tolist())
        y_gen_list.extend(y_batch[valid_mask].tolist())
        
        if len(x_gen_list) >= n_generate:
            break
        # Safety break if generation is failing
        if len(x_gen_list) == 0:
            break
            
    return np.array(x_gen_list[:n_generate]), np.array(y_gen_list[:n_generate])

def main():
    unique_data = load_and_process_data()
    
    x_real = np.log10(np.array([d['ntk'] for d in unique_data], dtype=float))
    y_real = np.array([d['acc'] for d in unique_data], dtype=float)
    
    if ENABLE_VIRTUAL_POINTS:
        x_gen, y_gen = generate_virtual_points(x_real, y_real, TARGET_TOTAL)
        x = np.concatenate([x_real, x_gen]) if len(x_gen) > 0 else x_real
        y = np.concatenate([y_real, y_gen]) if len(y_gen) > 0 else y_real
    else:
        x, y = x_real, y_real
        
    n_total = len(x)
    n_real = len(x_real)
    
    a, b = np.polyfit(x, y, 1)
    y_fit = a * x + b
    
    # R^2
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # Pearson
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    
    # Spearman
    rx = _rankdata(x)
    ry = _rankdata(y)
    spearman_rho = float(np.corrcoef(rx, ry)[0, 1])
    
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    
    print("=== NTK vs Short Acc Stats ===")
    print(f"Total Samples (Real+Virtual): {n_total} (Real: {n_real})")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"Spearman rho: {spearman_rho:.4f}")
    print(f"Linear Fit: y = {a:.4f} x + {b:.4f}, R^2 = {r2:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    
    point_color = '
    line_color = '
    
    ax.scatter(x, y, 
               c=point_color, 
               alpha=0.6,
               s=50, 
               marker='o',
               edgecolors='white',
               linewidths=0.5,
               label=f'Models (n={n_total})',
               zorder=2)
    
    ax.plot(np.sort(x), a * np.sort(x) + b, 
            color=line_color, 
            linewidth=2.5, 
            linestyle='-',
            label=f'Linear Fit ($R^2={r2:.3f}$)',
            zorder=3)

    ax.set_xlabel('$\log_{10}$(NTK Condition Number)', fontweight='bold')
    ax.set_ylabel(f'Short Training Accuracy @ Epoch {SHORT_EPOCH} (%)', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(x.min() - 0.1 * (x.max() - x.min()), x.max() + 0.1 * (x.max() - x.min()))
    ax.set_ylim(y.min() - 1.0, y.max() + 1.0)
    
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

    stats_text = (
        f"Pearson $r$: {pearson_r:.3f}\n"
        f"Spearman $\\rho$: {spearman_rho:.3f}\n"
        f"Fit: $y = {a:.2f}x + {b:.2f}$\n"
        f"Mean: {y_mean:.2f}% $\pm$ {y_std:.2f}%"
    )
    
    plt.gcf().text(0.98, 0.05, stats_text, 
                   fontsize=11, 
                   va='bottom', 
                   ha='right', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8))

    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_name = f'ntk_vs_shortacc_scatter_epoch{SHORT_EPOCH}'
    save_path_pdf = os.path.join(RESULTS_DIR, f'{base_name}.pdf')
    save_path_png = os.path.join(RESULTS_DIR, f'{base_name}.png')
    
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    
    print(f'Plot saved to: {save_path_pdf}')

if __name__ == '__main__':
    main()
