# -*- coding: utf-8 -*-
"""
Plot evolution vs random search comparison.
"""
import os
import json
import argparse
from utils.plotting import plot_comparison

def load_data(json_path: str):
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

def main():
    parser = argparse.ArgumentParser(description='Visualize Aging Evolution vs Random Search')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the JSON log file (e.g., comparison_curves_xxx.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: same dir as json)')
    
    args = parser.parse_args()
    
    print(f"INFO: loading_data={args.json_path}")
    evo_curve, rand_curve, evo_all, rand_all = load_data(args.json_path)
    
    # Calculate some basic stats for printing
    min_evo = min([v for v in evo_curve if v < 100000]) if any(v < 100000 for v in evo_curve) else 999999
    min_rand = min([v for v in rand_curve if v < 100000]) if any(v < 100000 for v in rand_curve) else 999999
    
    improvement = (min_rand - min_evo) / min_rand * 100 if min_rand > 0 else 0
    print(
        f"INFO: ntk_best evolution={min_evo:.2f} random={min_rand:.2f} "
        f"improvement={improvement:.2f}%"
    )
    
    if args.output is None:
        json_dir = os.path.dirname(args.json_path)
        json_name = os.path.splitext(os.path.basename(args.json_path))[0]
        args.output = os.path.join(json_dir, f"{json_name}_vis.png")
    
    plot_comparison(evo_curve, rand_curve, evo_all, rand_all, args.output)

if __name__ == '__main__':
    main()
