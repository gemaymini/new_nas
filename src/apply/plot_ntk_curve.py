# -*- coding: utf-8 -*-
"""
Plot NTK history curves from logs or checkpoints.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_ntk_history_from_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    if 'ntk_history' in checkpoint:
        return checkpoint['ntk_history']
    
    history = checkpoint.get('history', [])
    ntk_history = []
    for i, ind in enumerate(history):
        if hasattr(ind, 'fitness') and ind.fitness is not None:
            ntk_history.append((0, ind.id if hasattr(ind, 'id') else i, ind.fitness, ind.encoding))
    
    return ntk_history

def load_ntk_history_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ntk_history = []
    for item in data:
        ntk_history.append((
            item.get('step', 0),
            item.get('individual_id', 0),
            item.get('ntk'),
            item.get('encoding', [])
        ))
    
    return ntk_history

def plot_ntk_curve(ntk_history, output_path='ntk_curve.png', title_prefix=''):
    if not ntk_history:
        print("No NTK history to plot!")
        return
    
    steps = []
    ntk_values = []
    individual_ids = []
    
    for step, ind_id, ntk, encoding in ntk_history:
        if ntk is not None and ntk < 100000:
            steps.append(step)
            ntk_values.append(ntk)
            individual_ids.append(ind_id)
    
    if not steps:
        print("No valid NTK values to plot!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}NTK Analysis During Search', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    scatter = ax1.scatter(individual_ids, ntk_values, alpha=0.4, s=15, c=steps, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='Step')
    ax1.set_xlabel('Individual ID')
    ax1.set_ylabel('NTK Condition Number')
    ax1.set_title('All Individuals NTK (colored by step)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.scatter(steps, ntk_values, alpha=0.3, s=10, c='blue')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('NTK Condition Number')
    ax2.set_title('NTK vs Step')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    window_size = max(10, len(ntk_values) // 50)
    if len(ntk_values) >= window_size:
        moving_avg = []
        moving_std = []
        for i in range(len(ntk_values) - window_size + 1):
            window = ntk_values[i:i+window_size]
            moving_avg.append(np.mean(window))
            moving_std.append(np.std(window))
        moving_avg_steps = individual_ids[window_size-1:]
        
        ax3.fill_between(moving_avg_steps, 
                        np.array(moving_avg) - np.array(moving_std),
                        np.array(moving_avg) + np.array(moving_std),
                        alpha=0.2, color='blue', label='±1 std')
        ax3.plot(moving_avg_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (w={window_size})')
        ax3.scatter(individual_ids, ntk_values, alpha=0.1, s=3, c='blue')
        ax3.legend()
    else:
        ax3.scatter(individual_ids, ntk_values, alpha=0.5, s=10, c='blue')
    ax3.set_xlabel('Individual ID')
    ax3.set_ylabel('NTK Condition Number')
    ax3.set_title('NTK with Moving Average')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    cumulative_best = []
    current_best = float('inf')
    for ntk in ntk_values:
        current_best = min(current_best, ntk)
        cumulative_best.append(current_best)
    
    ax4.plot(individual_ids, ntk_values, 'b-', alpha=0.3, linewidth=0.5, label='Individual NTK')
    ax4.plot(individual_ids, cumulative_best, 'r-', linewidth=2, label='Cumulative Best')
    ax4.set_xlabel('Individual ID')
    ax4.set_ylabel('NTK Condition Number')
    ax4.set_title('Best NTK Progress')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    n, bins, patches = ax5.hist(ntk_values, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    
    mu, sigma = np.mean(ntk_values), np.std(ntk_values)
    x = np.linspace(min(ntk_values), max(ntk_values), 100)
    ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
    
    ax5.axvline(min(ntk_values), color='g', linestyle='--', linewidth=2, label=f'Best: {min(ntk_values):.2f}')
    ax5.set_xlabel('NTK Condition Number')
    ax5.set_ylabel('Density')
    ax5.set_title('NTK Distribution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    sorted_ntk = sorted(zip(ntk_values, individual_ids))
    top_k_values = [5, 10, 20, 50, 100]
    top_k_results = []
    
    for k in top_k_values:
        if k <= len(sorted_ntk):
            top_k_ntks = [x[0] for x in sorted_ntk[:k]]
            top_k_results.append({
                'k': k,
                'best': min(top_k_ntks),
                'mean': np.mean(top_k_ntks),
                'worst': max(top_k_ntks)
            })
    
    if top_k_results:
        k_values = [r['k'] for r in top_k_results]
        means = [r['mean'] for r in top_k_results]
        bests = [r['best'] for r in top_k_results]
        worsts = [r['worst'] for r in top_k_results]
        
        ax6.plot(k_values, means, 'bo-', linewidth=2, markersize=8, label='Mean of Top-K')
        ax6.fill_between(k_values, bests, worsts, alpha=0.3, color='blue', label='Min-Max Range')
        ax6.plot(k_values, bests, 'g--', linewidth=1, alpha=0.7, label='Best in Top-K')
        ax6.set_xlabel('K (Top-K)')
        ax6.set_ylabel('NTK Condition Number')
        ax6.set_title('Top-K NTK Analysis')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"NTK curve saved to {output_path}")
    
    print("\n" + "="*60)
    print("NTK Statistics Summary")
    print("="*60)
    print(f"Total Individuals: {len(ntk_values)}")
    print(f"Best NTK:          {min(ntk_values):.4f}")
    print(f"Mean NTK:          {np.mean(ntk_values):.4f}")
    print(f"Median NTK:        {np.median(ntk_values):.4f}")
    print(f"Std NTK:           {np.std(ntk_values):.4f}")
    print(f"Worst NTK:         {max(ntk_values):.4f}")
    
    print("\n" + "-"*60)
    print("Top 10 Best Individuals:")
    print("-"*60)
    for i, (ntk, ind_id) in enumerate(sorted_ntk[:10]):
        for step, id_, n, enc in ntk_history:
            if id_ == ind_id and n == ntk:
                print(f"  {i+1}. ID={ind_id:4d}, NTK={ntk:.4f}, Step={step}")
                break
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot NTK curve from checkpoint or JSON')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint .pkl file')
    parser.add_argument('--json', type=str, help='Path to ntk_history.json file')
    parser.add_argument('--output', type=str, default='ntk_curve.png', help='Output image path')
    parser.add_argument('--title', type=str, default='', help='Title prefix for the plot')
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.json:
        default_json = 'logs/ntk_history.json'
        if os.path.exists(default_json):
            args.json = default_json
            print(f"Using default JSON: {default_json}")
        else:
            checkpoint_dir = 'checkpoints'
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                    args.checkpoint = os.path.join(checkpoint_dir, files[0])
                    print(f"Using latest checkpoint: {args.checkpoint}")
    
    if not args.checkpoint and not args.json:
        print("Error: Please provide --checkpoint or --json file path")
        print("Usage examples:")
        print("  python src/apply/plot_ntk_curve.py --checkpoint checkpoints/checkpoint_step100.pkl")
        print("  python src/apply/plot_ntk_curve.py --json logs/ntk_history.json")
        sys.exit(1)
    
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        ntk_history = load_ntk_history_from_checkpoint(args.checkpoint)
    else:
        print(f"Loading from JSON: {args.json}")
        ntk_history = load_ntk_history_from_json(args.json)
    
    print(f"Loaded {len(ntk_history)} NTK records")
    
    plot_ntk_curve(ntk_history, args.output, args.title)

if __name__ == '__main__':
    main()
