# -*- coding: utf-8 -*-
"""
Plot algorithm comparison figures.
"""


import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime

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

ALGORITHM_ALIASES = {
    'Aging_NTK_Search': ['Aging_NTK_Search', 'three_stage_ea'],
    'Aging_Search': ['Aging_Search', 'traditional_ea'],
    'random_search': ['random_search']
}
ALGORITHM_KEYS = list(ALGORITHM_ALIASES.keys())

COLORS = {
    'Aging_NTK_Search': '#D62728', # Red (SOTA)
    'Aging_Search': '#1F77B4',     # Blue (Competitor)
    'random_search': '#7F7F7F'     # Grey (Baseline)
}

MARKERS = {
    'Aging_NTK_Search': 'D',       # Diamond
    'Aging_Search': 'o',           # Circle
    'random_search': 's'           # Square
}

LABELS = {
    'Aging_NTK_Search': 'Aging NTK Search',
    'Aging_Search': 'Aging Search',
    'random_search': 'Random Search'
}

def generate_simulated_data():
    np.random.seed(42)
    ts_params = np.linspace(0.8, 2.5, 15)
    ts_accs = 95.0 + 1.5 * np.exp(-ts_params) + np.random.normal(0, 0.1, 15)
    ts_accs = np.clip(ts_accs, 94.8, 96.6)
    
    te_params = np.linspace(2.0, 8.0, 20)
    te_accs = 92.0 - 1.0 * (te_params - 2.0) + np.random.normal(0, 0.8, 20)
    te_accs = np.clip(te_accs, 75.0, 93.0)
    
    rs_params = np.random.uniform(4.0, 14.0, 30)
    rs_accs = 85.0 - 2.0 * (rs_params - 4.0) + np.random.normal(0, 2.5, 30)
    rs_accs = np.clip(rs_accs, 40.0, 88.0)
    
    return {
        'Aging_NTK_Search': {'params': ts_params.tolist(), 'accs': ts_accs.tolist()},
        'Aging_Search': {'params': te_params.tolist(), 'accs': te_accs.tolist()},
        'random_search': {'params': rs_params.tolist(), 'accs': rs_accs.tolist()}
    }

def load_experiment_results(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for canonical_key, aliases in ALGORITHM_ALIASES.items():
        models = None
        for alias in aliases:
            if alias in data:
                models = data[alias]
                break
        if models:
            params = [m['param_count'] for m in models if m['param_count'] > 0]
            accs = [m['accuracy'] for m in models if m['accuracy'] > 0]
            result[canonical_key] = {'params': params, 'accs': accs}
        else:
            result[canonical_key] = {'params': [], 'accs': []}
    return result

def merge_experiment_results(results: list) -> dict:
    merged = {k: {'params': [], 'accs': []} for k in ALGORITHM_KEYS}
    for res in results:
        for key in merged.keys():
            if key in res:
                merged[key]['params'].extend(res[key].get('params', []))
                merged[key]['accs'].extend(res[key].get('accs', []))
    return merged

def load_experiment_results_from_dir(json_dir: str) -> dict:
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"Directory not found: {json_dir}")
    json_files = [os.path.join(json_dir, f) for f in sorted(os.listdir(json_dir)) if f.lower().endswith('.json')]
    results = []
    for path in json_files:
        try:
            results.append(load_experiment_results(path))
        except Exception:
            continue
    if not results:
        raise RuntimeError(f"No valid JSON found in {json_dir}")
    return merge_experiment_results(results)

def plot_algorithm_comparison(data: dict, 
                              output_path: str = None,
                              show_plot: bool = True,
                              show_inset: bool = True):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    plot_order = ['random_search', 'Aging_Search', 'Aging_NTK_Search']
    
    for key in plot_order:
        if key not in data or not data[key]['params']:
            continue
            
        params = np.array(data[key]['params'])
        accs = np.array(data[key]['accs'])
        
        ax.scatter(params, accs, 
                   c=COLORS[key], 
                   marker=MARKERS[key], 
                   s=60, 
                   alpha=0.8, 
                   edgecolors='white', 
                   linewidths=0.8,
                   label=LABELS[key],
                   zorder=3)
        
        if len(params) >= 3:
            try:
                points = np.column_stack([params, accs])
                hull = ConvexHull(points)
                hull_points = np.vstack((points[hull.vertices], points[hull.vertices[0]]))
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                        color=COLORS[key], 
                        alpha=0.15, 
                        zorder=2)
            except:
                pass

    ax.set_xlabel('Parameters (M)', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    all_p = [p for d in data.values() for p in d['params']]
    all_a = [a for d in data.values() for a in d['accs']]
    if all_p and all_a:
        ax.set_xlim(0, max(all_p) * 1.05)
        ax.set_ylim(min(50, min(all_a) * 0.9), min(100, max(all_a) * 1.02))

    ax.legend(loc='lower right', frameon=True, fancybox=False, 
              bbox_to_anchor=(0.98, 0.02), ncol=1)

    if show_inset and 'Aging_NTK_Search' in data and len(data['Aging_NTK_Search']['params']) >= 3:
        ts_p = np.array(data['Aging_NTK_Search']['params'])
        ts_a = np.array(data['Aging_NTK_Search']['accs'])
        
        x1, x2 = ts_p.min(), ts_p.max()
        y1, y2 = ts_a.min(), ts_a.max()
        x_range = x2 - x1
        y_range = y2 - y1
        rect_x1, rect_x2 = x1 - 0.1 * x_range, x2 + 0.1 * x_range
        rect_y1, rect_y2 = y1 - 0.2 * y_range, y2 + 0.2 * y_range
        
        axins = inset_axes(ax, width="30%", height="35%", loc='upper right')
        
        for sub_key in plot_order:
            if sub_key not in data or not data[sub_key]['params']: continue
            sub_p = np.array(data[sub_key]['params'])
            sub_a = np.array(data[sub_key]['accs'])
            
            axins.scatter(sub_p, sub_a, 
                          c=COLORS[sub_key], 
                          marker=MARKERS[sub_key], 
                          s=40, 
                          alpha=0.8,
                          edgecolors='white', linewidths=0.5,
                          zorder=3)
            if len(sub_p) >= 3:
                try:
                    points = np.column_stack([sub_p, sub_a])
                    hull = ConvexHull(points)
                    hull_pts = np.vstack((points[hull.vertices], points[hull.vertices[0]]))
                    axins.fill(hull_pts[:,0], hull_pts[:,1], color=COLORS[sub_key], alpha=0.15, zorder=2)
                except: pass

        axins.set_xlim(rect_x1, rect_x2)
        axins.set_ylim(rect_y1, rect_y2)
        axins.tick_params(axis='both', which='major', labelsize=10, width=1.0, length=3, direction='in')
        
        for spine in axins.spines.values():
            spine.set_edgecolor(COLORS['Aging_NTK_Search'])
            spine.set_linewidth(1.2)
            
        
        rect = Rectangle((rect_x1, rect_y1), rect_x2 - rect_x1, rect_y2 - rect_y1,
                         edgecolor=COLORS['Aging_NTK_Search'], 
                         facecolor='none', 
                         linestyle='--', 
                         linewidth=1.2,
                         zorder=10)
        ax.add_patch(rect)
        
        
        con_list = [
            ConnectionPatch(xyA=(rect_x1, rect_y2), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                           axesA=ax, axesB=axins, color="gray", linestyle="--", alpha=0.5, linewidth=1.0),
            ConnectionPatch(xyA=(rect_x2, rect_y1), xyB=(1, 0), coordsA="data", coordsB="axes fraction",
                           axesA=ax, axesB=axins, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
        ]
        
        for con in con_list:
            ax.add_artist(con)
            
    plt.tight_layout()
    
    if output_path:
        base_path = output_path.rsplit('.', 1)[0]
        plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(base_path + '.pdf', bbox_inches='tight')
        print(f"Saved to: {base_path}.pdf")
    
    if show_plot:
        plt.show()
    plt.close()

def print_statistics(data: dict):
    print("\n" + "=" * 60)
    print("                    Experiment Statistics")
    print("=" * 60)
    for key, label in LABELS.items():
        if key in data and len(data[key]['params']) > 0:
            params = np.array(data[key]['params'])
            accs = np.array(data[key]['accs'])
            print(f"\n{label}:")
            print(f"  Count: {len(params)}")
            print(f"  Acc: {accs.mean():.2f}% ± {accs.std():.2f}%")
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Generate Paper Plots')
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument('--json_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--use_simulated', action='store_true')
    parser.add_argument('--no_show', action='store_true')
    
    args = parser.parse_args()
    
    if args.json_dir:
        data = load_experiment_results_from_dir(args.json_dir)
    elif args.json_path and os.path.exists(args.json_path):
        data = load_experiment_results(args.json_path)
    elif args.use_simulated or (args.json_path is None and args.json_dir is None):
        print("Using simulated data...")
        data = generate_simulated_data()
    else:
        print("Error: Provide valid input.")
        return
    
    print_statistics(data)
    
    if args.output is None:
        output_dir = 'experiment_results'
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f'pareto_frontier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
    plot_algorithm_comparison(data, output_path=args.output, show_plot=not args.no_show)

if __name__ == '__main__':
    main()
