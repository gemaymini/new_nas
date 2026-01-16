# -*- coding: utf-8 -*-
"""
Shared plotting utilities for NAS experiments.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.spatial import ConvexHull
from utils.logger import logger

def plot_comparison(evolution_curve: List[float], random_curve: List[float], 
                    evolution_all_ntk: List[float], random_all_ntk: List[float],
                    output_path: str, title: str = None):
    """
    Plot comparison between Aging Evolution and Random Search.
    
    Args:
        evolution_curve: List of best-so-far NTK values for evolution.
        random_curve: List of best-so-far NTK values for random search.
        evolution_all_ntk: List of all individual NTK values for evolution.
        random_all_ntk: List of all individual NTK values for random search.
        output_path: Path to save the plot.
        title: Optional title for the plot.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    
    if title is None:
        title = f'Aging Evolution vs Random Search (N={len(evolution_curve)})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    steps = list(range(1, len(evolution_curve) + 1))
    
    # Filter valid values (assuming huge values are invalid/failures)
    evo_valid_vals = [v for v in evolution_all_ntk if v < 100000]
    rand_valid_vals = [v for v in random_all_ntk if v < 100000]
    
    # --- 1. Cumulative Best NTK ---
    ax1 = axes[0, 0]
    ax1.semilogy(steps, evolution_curve, 'b-', linewidth=2, label='Aging Evolution')
    ax1.semilogy(steps, random_curve, 'r-', linewidth=2, label='Random Search')
    ax1.set_xlabel('Evaluation Count')
    ax1.set_ylabel('Best NTK (log scale)')
    ax1.set_title('1. Cumulative Best NTK (Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.annotate(f'{evolution_curve[-1]:.1f}', xy=(len(steps), evolution_curve[-1]), 
                 xytext=(5, 0), textcoords='offset points', fontsize=9, color='blue')
    ax1.annotate(f'{random_curve[-1]:.1f}', xy=(len(steps), random_curve[-1]), 
                 xytext=(5, 0), textcoords='offset points', fontsize=9, color='red')
    
    # --- 2. All Individual NTK Values (Scatter) ---
    ax2 = axes[0, 1]
    evo_valid = [(i+1, v) for i, v in enumerate(evolution_all_ntk) if v < 100000]
    rand_valid = [(i+1, v) for i, v in enumerate(random_all_ntk) if v < 100000]
    if evo_valid:
        evo_steps, evo_vals = zip(*evo_valid)
        ax2.scatter(evo_steps, evo_vals, alpha=0.6, s=20, c='blue', label='Aging Evolution', edgecolors='none')
    if rand_valid:
        rand_steps, rand_vals = zip(*rand_valid)
        ax2.scatter(rand_steps, rand_vals, alpha=0.6, s=20, c='red', label='Random Search', edgecolors='none')
    ax2.set_yscale('log')
    ax2.set_xlabel('Evaluation Count')
    ax2.set_ylabel('Individual NTK Value (log scale)')
    ax2.set_title('2. All Individual NTK Values (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # --- 3. Cumulative Best NTK (Dual Y-Axis - Zoomed) ---
    ax3 = axes[1, 0]
    color1, color2 = 'blue', 'red'
    
    line1, = ax3.plot(steps, evolution_curve, color=color1, linewidth=2, label='Aging Evolution')
    ax3.set_xlabel('Evaluation Count')
    ax3.set_ylabel('Aging Evolution Best NTK', color=color1)
    ax3.tick_params(axis='y', labelcolor=color1)
    # Dynamic limits for better visibility
    if evolution_curve:
        ax3.set_ylim(min(evolution_curve) * 0.9, max(evolution_curve) * 1.1)
    
    ax3_twin = ax3.twinx()
    line2, = ax3_twin.plot(steps, random_curve, color=color2, linewidth=2, label='Random Search')
    ax3_twin.set_ylabel('Random Search Best NTK', color=color2)
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    if random_curve:
        ax3_twin.set_ylim(min(random_curve) * 0.9, max(random_curve) * 1.1)
    
    ax3.set_title('3. Cumulative Best NTK (Dual Y-Axis)')
    ax3.legend([line1, line2], ['Aging Evolution', 'Random Search'], loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # --- 4. NTK Distribution Histogram ---
    ax4 = axes[1, 1]
    all_valid = evo_valid_vals + rand_valid_vals
    if all_valid:
        log_min = np.log10(max(min(all_valid), 1))
        log_max = np.log10(max(all_valid))
        bins = np.logspace(log_min, log_max, 25)
        ax4.hist(evo_valid_vals, bins=bins, alpha=0.6, color='blue', label='Aging Evolution', edgecolor='black')
        ax4.hist(rand_valid_vals, bins=bins, alpha=0.6, color='red', label='Random Search', edgecolor='black')
        ax4.set_xscale('log')
    ax4.set_xlabel('NTK Condition Number (log scale)')
    ax4.set_ylabel('Count')
    ax4.set_title('4. NTK Distribution Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # --- 5. Relative Improvement Over Time ---
    ax5 = axes[2, 0]
    # Handle potential zero division if curves accidentally have 0
    safe_div = lambda x, y: (x - y) / x * 100 if x != 0 else 0
    
    if evolution_curve:
        evo_improvement = [safe_div(evolution_curve[0], v) for v in evolution_curve]
        ax5.plot(steps, evo_improvement, 'b-', linewidth=2, label='Aging Evolution')
        ax5.annotate(f'{evo_improvement[-1]:.1f}%', xy=(len(steps), evo_improvement[-1]), 
                     xytext=(5, 0), textcoords='offset points', fontsize=9, color='blue')

    if random_curve:
        rand_improvement = [safe_div(random_curve[0], v) for v in random_curve]
        ax5.plot(steps, rand_improvement, 'r-', linewidth=2, label='Random Search')
        ax5.annotate(f'{rand_improvement[-1]:.1f}%', xy=(len(steps), rand_improvement[-1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
                     
    ax5.set_xlabel('Evaluation Count')
    ax5.set_ylabel('Improvement from Initial (%)')
    ax5.set_title('5. Relative Improvement Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # --- 6. Moving Average NTK ---
    ax6 = axes[2, 1]
    window_size = max(5, len(evolution_all_ntk) // 20)
    
    def moving_average_continuous(data, window):
        cleaned = np.array([v if v < 100000 else np.nan for v in data], dtype=float)
        
        valid_mask = ~np.isnan(cleaned)
        if np.sum(valid_mask) >= 2:
            indices = np.arange(len(cleaned))
            cleaned = np.interp(indices, indices[valid_mask], cleaned[valid_mask])
        
        # Avoid crash on empty
        if len(cleaned) == 0:
            return []

        result = np.convolve(cleaned, np.ones(window)/window, mode='valid')
        # Pad beginning
        padding = [np.mean(cleaned[:i+1]) for i in range(window-1)]
        return padding + list(result)
    
    evo_ma = moving_average_continuous(evolution_all_ntk, window_size)
    rand_ma = moving_average_continuous(random_all_ntk, window_size)
    
    if len(evo_ma) == len(steps):
        ax6.semilogy(steps, evo_ma, 'b-', linewidth=2, label=f'Aging Evolution (MA={window_size})')
    if len(rand_ma) == len(steps):
        ax6.semilogy(steps, rand_ma, 'r-', linewidth=2, label=f'Random Search (MA={window_size})')
        
    ax6.set_xlabel('Evaluation Count')
    ax6.set_ylabel('Moving Average NTK (log scale)')
    ax6.set_title('6. Moving Average NTK')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')
    
    # --- 7. Sequential NTK Line Plot ---
    ax7 = axes[3, 0]
    evo_line = np.array([v if v < 100000 else np.nan for v in evolution_all_ntk], dtype=float)
    rand_line = np.array([v if v < 100000 else np.nan for v in random_all_ntk], dtype=float)
    
    for arr in [evo_line, rand_line]:
        valid_mask = ~np.isnan(arr)
        if np.sum(valid_mask) >= 2:
            indices = np.arange(len(arr))
            arr[:] = np.interp(indices, indices[valid_mask], arr[valid_mask])
    
    ax7.semilogy(steps, evo_line, 'b-', linewidth=1, alpha=0.8, label='Aging Evolution')
    ax7.semilogy(steps, rand_line, 'r-', linewidth=1, alpha=0.8, label='Random Search')
    ax7.set_xlabel('Evaluation Count')
    ax7.set_ylabel('NTK Condition Number (log)')
    ax7.set_title('7. Sequential NTK Line Plot')
    ax7.legend()
    ax7.grid(True, alpha=0.3, which='both')
    
    # --- 8. Summary Statistics Table ---
    ax8 = axes[3, 1]
    ax8.axis('off')
    
    table_data = [
        ['Metric', 'Aging Evolution', 'Random Search'],
        ['Best NTK', f'{min(evo_valid_vals):.2f}' if evo_valid_vals else 'N/A', 
         f'{min(rand_valid_vals):.2f}' if rand_valid_vals else 'N/A'],
        ['Mean NTK', f'{np.mean(evo_valid_vals):.2f}' if evo_valid_vals else 'N/A',
         f'{np.mean(rand_valid_vals):.2f}' if rand_valid_vals else 'N/A'],
        ['Std NTK', f'{np.std(evo_valid_vals):.2f}' if evo_valid_vals else 'N/A',
         f'{np.std(rand_valid_vals):.2f}' if rand_valid_vals else 'N/A'],
        ['Median NTK', f'{np.median(evo_valid_vals):.2f}' if evo_valid_vals else 'N/A',
         f'{np.median(rand_valid_vals):.2f}' if rand_valid_vals else 'N/A'],
        ['Valid Count', f'{len(evo_valid_vals)}/{len(evolution_all_ntk)}',
         f'{len(rand_valid_vals)}/{len(random_all_ntk)}'],
        ['Final Best', f'{evolution_curve[-1]:.2f}' if evolution_curve else 'N/A', 
         f'{random_curve[-1]:.2f}' if random_curve else 'N/A'],
    ]
    
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.35, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax8.set_title('8. Summary Statistics')
    

def plot_pareto_comparison(three_stage_models: List,
                           traditional_models: List,
                           random_models: List,
                           output_dir: str = None,
                           show_plot: bool = True):
    """
    Plot Pareto front comparison (Accuracy vs Parameters).
    
    Args:
        three_stage_models: List of model info objects/dicts.
        traditional_models: List of model info objects/dicts.
        random_models: List of model info objects/dicts.
        output_dir: Directory to save plots.
        show_plot: Whether to display plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def extract_data(models):
        # Handle both ModelInfo objects and dicts if necessary, assuming ModelInfo for now based on usage
        params = []
        accs = []
        for m in models:
            # Check if dict or object
            p = m.param_count if hasattr(m, 'param_count') else m.get('param_count', 0)
            a = m.accuracy if hasattr(m, 'accuracy') else m.get('accuracy', 0)
            if p > 0 and a > 0:
                params.append(p)
                accs.append(a)
        return np.array(params), np.array(accs)
    
    ts_params, ts_accs = extract_data(three_stage_models)
    te_params, te_accs = extract_data(traditional_models)
    rs_params, rs_accs = extract_data(random_models)
    
    colors = {
        'three_stage': '#D62728',
        'traditional': '#1F77B4',
        'random': '#7F7F7F'
    }
    
    alpha_hull = 0.25
    alpha_scatter = 0.8
    
    def plot_with_hull(params, accs, color, label, marker='o'):
        if len(params) < 3:
            ax.scatter(params, accs, c=color, label=label, s=80, 
                      alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
            return
        
        ax.scatter(params, accs, c=color, label=label, s=80, 
                  alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
        
        try:
            points = np.column_stack([params, accs])
            hull = ConvexHull(points)
            
            hull_points = points[hull.vertices]
            # Close loop
            hull_points = np.vstack([hull_points, hull_points[0]])
            
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=color, alpha=alpha_hull)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 
                   color=color, linewidth=2, alpha=0.7)
        except Exception as e:
            logger.warning(f"Cannot draw convex hull: {e}")
    
    if len(ts_params) > 0:
        plot_with_hull(ts_params, ts_accs, colors['three_stage'], 'Three-Stage EA', 'o')
    if len(te_params) > 0:
        plot_with_hull(te_params, te_accs, colors['traditional'], 'Traditional EA', 's')
    if len(rs_params) > 0:
        plot_with_hull(rs_params, rs_accs, colors['random'], 'Random Search', '^')
    
    ax.set_xlabel('Parameters (M)', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Comparison of Three Search Algorithms', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    stats_text = []
    if len(ts_accs) > 0:
        stats_text.append(f"Three-Stage EA: Avg Acc={np.mean(ts_accs):.2f}%, Avg Params={np.mean(ts_params):.2f}M")
    if len(te_accs) > 0:
        stats_text.append(f"Traditional EA: Avg Acc={np.mean(te_accs):.2f}%, Avg Params={np.mean(te_params):.2f}M")
    if len(rs_accs) > 0:
        stats_text.append(f"Random Search: Avg Acc={np.mean(rs_accs):.2f}%, Avg Params={np.mean(rs_params):.2f}M")
    
    stats_str = '\n'.join(stats_text)
    ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'algorithm_comparison_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chart saved to: {output_path}")
        
        pdf_path = os.path.join(output_dir, f'algorithm_comparison_{timestamp}.pdf')
        plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
        logger.info(f"PDF saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
