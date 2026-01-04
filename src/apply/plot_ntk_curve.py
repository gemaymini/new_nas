# -*- coding: utf-8 -*-
"""
从checkpoint或JSON文件绘制NTK曲线

用法:
    python src/apply/plot_ntk_curve.py --checkpoint checkpoints/checkpoint_step100.pkl
    python src/apply/plot_ntk_curve.py --json logs/ntk_history.json
    python src/apply/plot_ntk_curve.py --checkpoint checkpoints/checkpoint_step100.pkl --output ntk_analysis.png
"""
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_ntk_history_from_checkpoint(checkpoint_path):
    """从checkpoint文件加载NTK历史"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # 新版checkpoint包含ntk_history
    if 'ntk_history' in checkpoint:
        return checkpoint['ntk_history']
    
    # 旧版checkpoint需要从history中提取
    history = checkpoint.get('history', [])
    ntk_history = []
    for i, ind in enumerate(history):
        # 优先使用 ntk_score（原始NTK值），如果没有则使用 fitness
        ntk_value = getattr(ind, 'ntk_score', None)
        if ntk_value is None:
            ntk_value = getattr(ind, 'fitness', None)
        if ntk_value is not None:
            ntk_history.append((0, ind.id if hasattr(ind, 'id') else i, ntk_value, ind.encoding))
    
    return ntk_history

def load_ntk_history_from_json(json_path):
    """从JSON文件加载NTK历史（支持新旧格式）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ntk_history = []
    for item in data:
        # 新格式包含 fitness 和 param_count
        if 'fitness' in item:
            ntk_history.append((
                item.get('step', 0),
                item.get('individual_id', 0),
                item.get('ntk'),
                item.get('fitness'),
                item.get('param_count'),
                item.get('encoding', [])
            ))
        else:
            # 旧格式：4元素
            ntk_history.append((
                item.get('step', 0),
                item.get('individual_id', 0),
                item.get('ntk'),
                item.get('encoding', [])
            ))
    
    return ntk_history

def plot_ntk_curve(ntk_history, output_path='ntk_curve.png', title_prefix=''):
    """
    绘制搜索过程中NTK值的变化曲线
    """
    if not ntk_history:
        print("No NTK history to plot!")
        return
    
    # 提取有效数据（兼容新旧格式）
    steps = []
    ntk_values = []
    individual_ids = []
    
    for record in ntk_history:
        if len(record) == 4:  # 旧格式
            step, ind_id, ntk, encoding = record
        else:  # 新格式（6元素）
            step, ind_id, ntk, fitness, param_count, encoding = record
        
        if ntk is not None and ntk < 100000:  # 排除无效值
            steps.append(step)
            ntk_values.append(ntk)
            individual_ids.append(ind_id)
        if ntk is not None and ntk < 100000:  # 排除无效值
            steps.append(step)
            ntk_values.append(ntk)
            individual_ids.append(ind_id)
    
    if not steps:
        print("No valid NTK values to plot!")
        return
    
    # 创建图形 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}NTK Analysis During Search', fontsize=14, fontweight='bold')
    
    # 1. 所有个体的NTK散点图（按个体ID）
    ax1 = axes[0, 0]
    scatter = ax1.scatter(individual_ids, ntk_values, alpha=0.4, s=15, c=steps, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='Step')
    ax1.set_xlabel('Individual ID')
    ax1.set_ylabel('NTK Condition Number')
    ax1.set_title('All Individuals NTK (colored by step)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 按Step的NTK散点图
    ax2 = axes[0, 1]
    ax2.scatter(steps, ntk_values, alpha=0.3, s=10, c='blue')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('NTK Condition Number')
    ax2.set_title('NTK vs Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. 滑动窗口平均NTK曲线
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
    
    # 4. 累积最佳NTK曲线
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
    
    # 5. NTK分布直方图
    ax5 = axes[1, 1]
    n, bins, patches = ax5.hist(ntk_values, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    
    # 拟合正态分布
    mu, sigma = np.mean(ntk_values), np.std(ntk_values)
    x = np.linspace(min(ntk_values), max(ntk_values), 100)
    ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
    
    ax5.axvline(min(ntk_values), color='g', linestyle='--', linewidth=2, label=f'Best: {min(ntk_values):.2f}')
    ax5.set_xlabel('NTK Condition Number')
    ax5.set_ylabel('Density')
    ax5.set_title('NTK Distribution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Top-K分析
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
    
    # 打印详细统计信息
    print("\n" + "="*60)
    print("NTK Statistics Summary")
    print("="*60)
    print(f"Total Individuals: {len(ntk_values)}")
    print(f"Best NTK:          {min(ntk_values):.4f}")
    print(f"Mean NTK:          {np.mean(ntk_values):.4f}")
    print(f"Median NTK:        {np.median(ntk_values):.4f}")
    print(f"Std NTK:           {np.std(ntk_values):.4f}")
    print(f"Worst NTK:         {max(ntk_values):.4f}")
    
    # Top-10 最佳个体
    print("\n" + "-"*60)
    print("Top 10 Best Individuals:")
    print("-"*60)
    for i, (ntk, ind_id) in enumerate(sorted_ntk[:10]):
        # 找到对应的encoding（兼容新旧格式）
        for record in ntk_history:
            if len(record) == 4:
                step, id_, n, enc = record
            else:
                step, id_, n, fitness, param_count, enc = record
            if id_ == ind_id and n == ntk:
                print(f"  {i+1}. ID={ind_id:4d}, NTK={ntk:.4f}, Step={step}")
                break
    print("="*60)


def plot_fitness_curve(ntk_history, output_path='fitness_curve.png', title_prefix=''):
    """
    绘制搜索过程中Fitness值的变化曲线
    仅支持新格式（包含fitness字段）
    """
    if not ntk_history:
        print("No history to plot!")
        return
    
    # 提取有效数据（仅新格式）
    steps = []
    fitness_values = []
    ntk_values = []
    param_counts = []
    individual_ids = []
    
    for record in ntk_history:
        if len(record) == 6:  # 新格式
            step, ind_id, ntk, fitness, param_count, encoding = record
            if fitness is not None and fitness < 1.0:
                steps.append(step)
                fitness_values.append(fitness)
                individual_ids.append(ind_id)
                if ntk is not None:
                    ntk_values.append(ntk)
                if param_count is not None:
                    param_counts.append(param_count / 1e6)  # 转为百万
    
    if not steps:
        print("No valid Fitness values to plot! (需要新格式的历史记录)")
        return
    
    # 创建图形 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}Fitness Analysis During Search', fontsize=14, fontweight='bold')
    
    # 1. 所有个体的Fitness散点图（按个体ID）
    ax1 = axes[0, 0]
    scatter = ax1.scatter(individual_ids, fitness_values, alpha=0.4, s=15, c=steps, cmap='viridis')
    plt.colorbar(scatter, ax=ax1, label='Step')
    ax1.set_xlabel('Individual ID')
    ax1.set_ylabel('Fitness (lower is better)')
    ax1.set_title('All Individuals Fitness (colored by step)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 按Step的Fitness散点图
    ax2 = axes[0, 1]
    ax2.scatter(steps, fitness_values, alpha=0.3, s=10, c='purple')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Fitness (lower is better)')
    ax2.set_title('Fitness vs Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. 滑动窗口平均Fitness曲线
    ax3 = axes[0, 2]
    window_size = max(10, len(fitness_values) // 50)
    if len(fitness_values) >= window_size:
        moving_avg = []
        for i in range(len(fitness_values) - window_size + 1):
            avg = sum(fitness_values[i:i+window_size]) / window_size
            moving_avg.append(avg)
        moving_avg_steps = steps[window_size-1:]
        ax3.plot(moving_avg_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (w={window_size})')
        ax3.scatter(steps, fitness_values, alpha=0.2, s=5, c='purple', label='Individual')
        ax3.legend()
    else:
        ax3.scatter(steps, fitness_values, alpha=0.5, s=10, c='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Fitness (lower is better)')
    ax3.set_title('Fitness with Moving Average')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积最佳Fitness曲线
    ax4 = axes[1, 0]
    step_best = {}
    for record in ntk_history:
        if len(record) == 6:
            step, ind_id, ntk, fitness, param_count, encoding = record
            if fitness is not None and fitness < 1.0:
                if step not in step_best or fitness < step_best[step]:
                    step_best[step] = fitness
    
    sorted_steps = sorted(step_best.keys())
    best_per_step = [step_best[s] for s in sorted_steps]
    
    cumulative_best = []
    current_best = float('inf')
    for f in best_per_step:
        current_best = min(current_best, f)
        cumulative_best.append(current_best)
    
    ax4.plot(sorted_steps, best_per_step, 'g-', alpha=0.5, label='Best per Step')
    ax4.plot(sorted_steps, cumulative_best, 'r-', linewidth=2, label='Cumulative Best')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Fitness (lower is better)')
    ax4.set_title('Best Fitness Progress')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Fitness分布直方图
    ax5 = axes[1, 1]
    ax5.hist(fitness_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
    best_fitness = min(fitness_values)
    mean_fitness = np.mean(fitness_values)
    ax5.axvline(best_fitness, color='r', linestyle='--', linewidth=2, label=f'Best: {best_fitness:.4f}')
    ax5.axvline(mean_fitness, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean_fitness:.4f}')
    ax5.set_xlabel('Fitness (lower is better)')
    ax5.set_ylabel('Count')
    ax5.set_title('Fitness Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 参数量 vs Fitness 相关性（如果有参数量数据）
    ax6 = axes[1, 2]
    if len(param_counts) == len(fitness_values) and param_counts:
        scatter6 = ax6.scatter(param_counts, fitness_values, alpha=0.5, s=15, 
                               c=ntk_values if len(ntk_values) == len(param_counts) else 'purple', 
                               cmap='viridis')
        ax6.set_xlabel('Parameter Count (M)')
        ax6.set_ylabel('Fitness (lower is better)')
        ax6.set_title('Params vs Fitness (color=NTK)')
        if len(ntk_values) == len(param_counts):
            plt.colorbar(scatter6, ax=ax6, label='NTK')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No param data available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Params vs Fitness')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Fitness curve saved to {output_path}")
    
    # 打印详细统计信息
    print("\n" + "="*60)
    print("Fitness Statistics Summary")
    print("="*60)
    print(f"Total Individuals: {len(fitness_values)}")
    print(f"Best Fitness:      {min(fitness_values):.6f}")
    print(f"Mean Fitness:      {np.mean(fitness_values):.6f}")
    print(f"Median Fitness:    {np.median(fitness_values):.6f}")
    print(f"Std Fitness:       {np.std(fitness_values):.6f}")
    print(f"Worst Fitness:     {max(fitness_values):.6f}")
    
    # Top-10 最佳个体
    sorted_fitness = sorted(zip(fitness_values, individual_ids))
    print("\n" + "-"*60)
    print("Top 10 Best Individuals by Fitness:")
    print("-"*60)
    for i, (fitness, ind_id) in enumerate(sorted_fitness[:10]):
        for record in ntk_history:
            if len(record) == 6:
                step, id_, ntk, fit, param, enc = record
                if id_ == ind_id and fit == fitness:
                    param_str = f", Params={param/1e6:.2f}M" if param else ""
                    ntk_str = f", NTK={ntk:.2f}" if ntk else ""
                    print(f"  {i+1}. ID={ind_id:4d}, Fitness={fitness:.6f}{ntk_str}{param_str}")
                    break
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot NTK/Fitness curve from checkpoint or JSON')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint .pkl file')
    parser.add_argument('--json', type=str, help='Path to ntk_history.json file')
    parser.add_argument('--output', type=str, default='ntk_curve.png', help='Output image path')
    parser.add_argument('--title', type=str, default='', help='Title prefix for the plot')
    parser.add_argument('--fitness', action='store_true', help='Also plot fitness curve')
    parser.add_argument('--fitness-only', action='store_true', help='Only plot fitness curve')
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.json:
        # 默认查找logs/ntk_history.json
        default_json = 'logs/ntk_history.json'
        if os.path.exists(default_json):
            args.json = default_json
            print(f"Using default JSON: {default_json}")
        else:
            # 查找最新的checkpoint
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
        print("  python src/apply/plot_ntk_curve.py --json logs/ntk_history.json --fitness")
        print("  python src/apply/plot_ntk_curve.py --json logs/ntk_history.json --fitness-only")
        sys.exit(1)
    
    # 加载数据
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        ntk_history = load_ntk_history_from_checkpoint(args.checkpoint)
    else:
        print(f"Loading from JSON: {args.json}")
        ntk_history = load_ntk_history_from_json(args.json)
    
    print(f"Loaded {len(ntk_history)} records")
    
    # 检查是否有新格式数据
    has_fitness = any(len(record) == 6 for record in ntk_history)
    if has_fitness:
        print("Detected new format with fitness data")
    
    # 绘制曲线
    if not args.fitness_only:
        plot_ntk_curve(ntk_history, args.output, args.title)
    
    if args.fitness or args.fitness_only:
        if has_fitness:
            fitness_output = args.output.replace('.png', '_fitness.png') if not args.fitness_only else args.output
            if args.fitness_only:
                fitness_output = args.output.replace('ntk_curve', 'fitness_curve') if 'ntk_curve' in args.output else args.output
            plot_fitness_curve(ntk_history, fitness_output, args.title)
        else:
            print("Warning: No fitness data found (old format). Cannot plot fitness curve.")

if __name__ == '__main__':
    main()
