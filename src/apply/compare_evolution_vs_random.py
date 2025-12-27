# -*- coding: utf-8 -*-
"""
比较老化进化算法 (Aging Evolution) 和随机搜索 (Random Search) 的性能差异

实验设计：
1. 两种算法使用相同的评估次数（NTK评估）
2. 记录每步的最佳NTK值变化曲线
3. 统计最终结果并绘制对比图
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, clear_gpu_memory
from utils.logger import logger


class RandomSearch:
    """随机搜索算法"""
    
    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations
        self.history: List[Individual] = []
        self.best_fitness_curve: List[float] = []  # 记录每步的累积最佳fitness
        self.all_fitness_values: List[float] = []  # 记录每个个体的NTK值
        
    def run(self) -> Tuple[List[Individual], List[float], List[float]]:
        """
        运行随机搜索
        
        Returns:
            history: 所有评估过的个体
            best_fitness_curve: 每步的累积最佳fitness
            all_fitness_values: 每个个体的NTK值
        """
        logger.info(f"Starting Random Search for {self.max_evaluations} evaluations...")
        
        best_fitness = float('inf')
        
        for i in range(self.max_evaluations):
            # 随机生成一个有效个体
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            # 评估NTK
            fitness_evaluator.evaluate_individual(ind)
            self.history.append(ind)
            
            # 记录当前个体的NTK值
            current_fitness = ind.fitness if ind.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
            # 更新最佳fitness（越小越好）
            if ind.fitness is not None and ind.fitness < 100000:
                best_fitness = min(best_fitness, ind.fitness)
            
            self.best_fitness_curve.append(best_fitness if best_fitness < float('inf') else 100000)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Random Search [{i+1}/{self.max_evaluations}] Best NTK: {best_fitness:.4f}")
                clear_gpu_memory()
        
        logger.info(f"Random Search completed. Best NTK: {best_fitness:.4f}")
        return self.history, self.best_fitness_curve, self.all_fitness_values


class AgingEvolutionSearch:
    """老化进化算法（简化版，用于对比实验）"""
    
    def __init__(self, max_evaluations: int, population_size: int = None):
        self.max_evaluations = max_evaluations
        self.population_size = population_size or config.POPULATION_SIZE
        self.population = deque()
        self.history: List[Individual] = []
        self.best_fitness_curve: List[float] = []
        self.all_fitness_values: List[float] = []  # 记录每个个体的NTK值
        
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """锦标赛选择"""
        current_pop_list = list(self.population)
        parents = selection_operator.tournament_selection(
            current_pop_list,
            tournament_size=config.TOURNAMENT_SIZE,
            num_winners=config.TOURNAMENT_WINNERS
        )
        if len(parents) < 2:
            return parents[0], parents[0]
        return parents[0], parents[1]
    
    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        """生成后代"""
        child = None
        
        # 交叉
        if random.random() < config.PROB_CROSSOVER:
            c1, c2 = crossover_operator.crossover(parent1, parent2)
            child = random.choice([c1, c2])
        else:
            child = random.choice([parent1, parent2]).copy()
        
        # 变异
        if random.random() < config.PROB_MUTATION:
            child = mutation_operator.mutate(child)
        
        # 验证
        if not Encoder.validate_encoding(child.encoding):
            # 修复：重新变异父代
            for _ in range(20):
                child = mutation_operator.mutate(random.choice([parent1, parent2]))
                if Encoder.validate_encoding(child.encoding):
                    break
            else:
                child = random.choice([parent1, parent2]).copy()
        
        return child
    
    def run(self) -> Tuple[List[Individual], List[float], List[float]]:
        """
        运行老化进化搜索
        
        Returns:
            history: 所有评估过的个体
            best_fitness_curve: 每步的累积最佳fitness
            all_fitness_values: 每个个体的NTK值
        """
        logger.info(f"Starting Aging Evolution for {self.max_evaluations} evaluations...")
        logger.info(f"Population size: {self.population_size}")
        
        best_fitness = float('inf')
        eval_count = 0
        
        # 1. 初始化种群
        logger.info("Initializing population...")
        while len(self.population) < self.population_size and eval_count < self.max_evaluations:
            ind = population_initializer.create_valid_individual()
            ind.id = eval_count
            fitness_evaluator.evaluate_individual(ind)
            
            self.population.append(ind)
            self.history.append(ind)
            
            # 记录当前个体的NTK值
            current_fitness = ind.fitness if ind.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
            if ind.fitness is not None and ind.fitness < 100000:
                best_fitness = min(best_fitness, ind.fitness)
            
            self.best_fitness_curve.append(best_fitness if best_fitness < float('inf') else 100000)
            eval_count += 1
            
            if eval_count % 10 == 0:
                logger.info(f"Initialization [{eval_count}/{self.population_size}]")
        
        logger.info(f"Population initialized. Size: {len(self.population)}")
        
        # 2. 进化循环
        while eval_count < self.max_evaluations:
            # 选择父代
            parent1, parent2 = self._select_parents()
            
            # 生成后代
            child = self._generate_offspring(parent1, parent2)
            child.id = eval_count
            
            # 评估
            fitness_evaluator.evaluate_individual(child)
            
            # 更新种群（移除最老的，添加新的）
            self.population.popleft()
            self.population.append(child)
            self.history.append(child)
            
            # 记录当前个体的NTK值
            current_fitness = child.fitness if child.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
            # 更新最佳fitness
            if child.fitness is not None and child.fitness < 100000:
                best_fitness = min(best_fitness, child.fitness)
            
            self.best_fitness_curve.append(best_fitness if best_fitness < float('inf') else 100000)
            eval_count += 1
            
            if eval_count % 20 == 0:
                logger.info(f"Aging Evolution [{eval_count}/{self.max_evaluations}] Best NTK: {best_fitness:.4f}")
                clear_gpu_memory()
        
        logger.info(f"Aging Evolution completed. Best NTK: {best_fitness:.4f}")
        return self.history, self.best_fitness_curve, self.all_fitness_values


def plot_comparison(evolution_curve: List[float], random_curve: List[float], 
                    evolution_all_ntk: List[float], random_all_ntk: List[float],
                    output_path: str, title: str = None):
    """
    绘制对比图
    
    Args:
        evolution_curve: 老化进化的累积最佳NTK曲线
        random_curve: 随机搜索的累积最佳NTK曲线
        evolution_all_ntk: 老化进化搜索的每个个体NTK值
        random_all_ntk: 随机搜索的每个个体NTK值
        output_path: 输出路径
        title: 图表标题
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    if title is None:
        title = f'Aging Evolution vs Random Search (N={len(evolution_curve)})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    steps = list(range(1, len(evolution_curve) + 1))
    
    # 1. 累积最佳NTK曲线对比
    ax1 = axes[0, 0]
    ax1.plot(steps, evolution_curve, 'b-', linewidth=2, label='Aging Evolution')
    ax1.plot(steps, random_curve, 'r-', linewidth=2, label='Random Search')
    ax1.set_xlabel('Evaluation Count')
    ax1.set_ylabel('Best NTK Condition Number')
    ax1.set_title('Cumulative Best NTK (Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 每个个体NTK值散点图对比
    ax2 = axes[0, 1]
    # 过滤掉无效值 (100000)
    evo_valid = [(i+1, v) for i, v in enumerate(evolution_all_ntk) if v < 100000]
    rand_valid = [(i+1, v) for i, v in enumerate(random_all_ntk) if v < 100000]
    if evo_valid:
        evo_steps, evo_vals = zip(*evo_valid)
        ax2.scatter(evo_steps, evo_vals, alpha=0.5, s=15, c='blue', label='Aging Evolution')
    if rand_valid:
        rand_steps, rand_vals = zip(*rand_valid)
        ax2.scatter(rand_steps, rand_vals, alpha=0.5, s=15, c='red', label='Random Search')
    ax2.set_xlabel('Evaluation Count')
    ax2.set_ylabel('Individual NTK Value')
    ax2.set_title('All Individual NTK Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 收敛速度对比（对数坐标）
    ax3 = axes[1, 0]
    ax3.semilogy(steps, evolution_curve, 'b-', linewidth=2, label='Aging Evolution')
    ax3.semilogy(steps, random_curve, 'r-', linewidth=2, label='Random Search')
    ax3.set_xlabel('Evaluation Count')
    ax3.set_ylabel('Best NTK (log scale)')
    ax3.set_title('Convergence Speed (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. NTK分布直方图对比
    ax4 = axes[1, 1]
    evo_valid_vals = [v for v in evolution_all_ntk if v < 100000]
    rand_valid_vals = [v for v in random_all_ntk if v < 100000]
    
    # 计算合适的bin范围
    all_valid = evo_valid_vals + rand_valid_vals
    if all_valid:
        bin_min = min(all_valid)
        bin_max = max(all_valid)
        bins = np.linspace(bin_min, bin_max, 30)
        ax4.hist(evo_valid_vals, bins=bins, alpha=0.6, color='blue', label='Aging Evolution', edgecolor='black')
        ax4.hist(rand_valid_vals, bins=bins, alpha=0.6, color='red', label='Random Search', edgecolor='black')
    ax4.set_xlabel('NTK Condition Number')
    ax4.set_ylabel('Count')
    ax4.set_title('NTK Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 相对改进比例
    ax5 = axes[2, 0]
    # 计算每步相对于初始值的改进比例
    evo_improvement = [(evolution_curve[0] - v) / evolution_curve[0] * 100 for v in evolution_curve]
    rand_improvement = [(random_curve[0] - v) / random_curve[0] * 100 for v in random_curve]
    ax5.plot(steps, evo_improvement, 'b-', linewidth=2, label='Aging Evolution')
    ax5.plot(steps, rand_improvement, 'r-', linewidth=2, label='Random Search')
    ax5.set_xlabel('Evaluation Count')
    ax5.set_ylabel('Improvement from Initial (%)')
    ax5.set_title('Relative Improvement Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 滑动窗口平均NTK对比
    ax6 = axes[2, 1]
    window_size = max(5, len(evolution_all_ntk) // 20)
    
    def moving_average(data, window):
        """计算滑动窗口平均，跳过无效值"""
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_vals = [v for v in data[start:i+1] if v < 100000]
            if window_vals:
                result.append(np.mean(window_vals))
            else:
                result.append(np.nan)
        return result
    
    evo_ma = moving_average(evolution_all_ntk, window_size)
    rand_ma = moving_average(random_all_ntk, window_size)
    
    ax6.plot(steps, evo_ma, 'b-', linewidth=2, label=f'Aging Evolution (MA={window_size})')
    ax6.plot(steps, rand_ma, 'r-', linewidth=2, label=f'Random Search (MA={window_size})')
    ax6.set_xlabel('Evaluation Count')
    ax6.set_ylabel('Moving Average NTK')
    ax6.set_title('Moving Average NTK Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plot saved to {output_path}")


def compute_statistics(evolution_history: List[Individual], random_history: List[Individual],
                       evolution_curve: List[float], random_curve: List[float]) -> dict:
    """
    计算统计指标
    """
    # 提取有效的fitness值
    evo_fitnesses = [ind.fitness for ind in evolution_history 
                     if ind.fitness is not None and ind.fitness < 100000]
    rand_fitnesses = [ind.fitness for ind in random_history 
                      if ind.fitness is not None and ind.fitness < 100000]
    
    stats = {
        'evolution': {
            'total_evaluations': len(evolution_history),
            'valid_evaluations': len(evo_fitnesses),
            'best_ntk': float(min(evo_fitnesses)) if evo_fitnesses else None,
            'mean_ntk': float(np.mean(evo_fitnesses)) if evo_fitnesses else None,
            'std_ntk': float(np.std(evo_fitnesses)) if evo_fitnesses else None,
            'final_best': float(evolution_curve[-1]) if evolution_curve else None,
        },
        'random': {
            'total_evaluations': len(random_history),
            'valid_evaluations': len(rand_fitnesses),
            'best_ntk': float(min(rand_fitnesses)) if rand_fitnesses else None,
            'mean_ntk': float(np.mean(rand_fitnesses)) if rand_fitnesses else None,
            'std_ntk': float(np.std(rand_fitnesses)) if rand_fitnesses else None,
            'final_best': float(random_curve[-1]) if random_curve else None,
        },
        'comparison': {}
    }
    
    # 比较指标
    if stats['evolution']['best_ntk'] and stats['random']['best_ntk']:
        improvement = ((stats['random']['best_ntk'] - stats['evolution']['best_ntk']) 
                       / stats['random']['best_ntk'] * 100)
        stats['comparison']['best_ntk_improvement_%'] = round(float(improvement), 2)
        stats['comparison']['evolution_better'] = bool(stats['evolution']['best_ntk'] < stats['random']['best_ntk'])
    
    # 计算收敛速度（达到某个阈值所需的评估次数）
    if evolution_curve and random_curve:
        # 以随机搜索最终值为阈值
        threshold = random_curve[-1] * 1.05  # 略高于随机搜索最终值
        
        evo_converge_step = None
        for i, v in enumerate(evolution_curve):
            if v <= threshold:
                evo_converge_step = i + 1
                break
        
        rand_converge_step = len(random_curve)  # 随机搜索到最后才达到
        
        stats['comparison']['threshold'] = float(threshold)
        stats['comparison']['evolution_steps_to_threshold'] = evo_converge_step
        stats['comparison']['random_steps_to_threshold'] = rand_converge_step
        
        if evo_converge_step:
            speedup = rand_converge_step / evo_converge_step
            stats['comparison']['speedup'] = round(float(speedup), 2)
    
    return stats


def run_experiment(max_evaluations: int, population_size: int, seed: int = None, 
                   output_dir: str = None):
    """
    运行对比实验
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'experiment_results')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Aging Evolution vs Random Search Comparison Experiment")
    logger.info("=" * 60)
    logger.info(f"Max evaluations: {max_evaluations}")
    logger.info(f"Population size (for Evolution): {population_size}")
    
    # 1. 运行老化进化算法
    logger.info("\n" + "=" * 40)
    logger.info("Phase 1: Running Aging Evolution")
    logger.info("=" * 40)
    start_time = time.time()
    evolution_search = AgingEvolutionSearch(max_evaluations, population_size)
    evolution_history, evolution_curve, evolution_all_ntk = evolution_search.run()
    evolution_time = time.time() - start_time
    logger.info(f"Aging Evolution time: {evolution_time:.2f}s")
    
    clear_gpu_memory()
    
    # 2. 运行随机搜索
    logger.info("\n" + "=" * 40)
    logger.info("Phase 2: Running Random Search")
    logger.info("=" * 40)
    start_time = time.time()
    random_search = RandomSearch(max_evaluations)
    random_history, random_curve, random_all_ntk = random_search.run()
    random_time = time.time() - start_time
    logger.info(f"Random Search time: {random_time:.2f}s")
    
    clear_gpu_memory()
    
    # 3. 计算统计指标
    logger.info("\n" + "=" * 40)
    logger.info("Phase 3: Computing Statistics")
    logger.info("=" * 40)
    stats = compute_statistics(evolution_history, random_history, evolution_curve, random_curve)
    stats['timing'] = {
        'evolution_time_s': round(evolution_time, 2),
        'random_time_s': round(random_time, 2),
    }
    stats['config'] = {
        'max_evaluations': max_evaluations,
        'population_size': population_size,
        'seed': seed,
    }
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} {'Evolution':<15} {'Random':<15}")
    logger.info("-" * 60)
    logger.info(f"{'Best NTK':<30} {stats['evolution']['best_ntk']:<15.4f} {stats['random']['best_ntk']:<15.4f}")
    logger.info(f"{'Mean NTK':<30} {stats['evolution']['mean_ntk']:<15.4f} {stats['random']['mean_ntk']:<15.4f}")
    logger.info(f"{'Std NTK':<30} {stats['evolution']['std_ntk']:<15.4f} {stats['random']['std_ntk']:<15.4f}")
    logger.info(f"{'Valid Evaluations':<30} {stats['evolution']['valid_evaluations']:<15} {stats['random']['valid_evaluations']:<15}")
    logger.info("-" * 60)
    
    if 'best_ntk_improvement_%' in stats['comparison']:
        logger.info(f"Evolution Improvement: {stats['comparison']['best_ntk_improvement_%']:.2f}%")
    if 'speedup' in stats['comparison']:
        logger.info(f"Speedup: {stats['comparison']['speedup']:.2f}x")
    
    logger.info("=" * 60)
    
    # 4. 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存统计数据
    stats_path = os.path.join(output_dir, f'comparison_stats_{timestamp}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistics saved to {stats_path}")
    
    # 保存曲线数据
    curves_data = {
        'evolution_curve': evolution_curve,
        'random_curve': random_curve,
        'evolution_all_ntk': evolution_all_ntk,
        'random_all_ntk': random_all_ntk,
    }
    curves_path = os.path.join(output_dir, f'comparison_curves_{timestamp}.json')
    with open(curves_path, 'w', encoding='utf-8') as f:
        json.dump(curves_data, f, indent=2)
    logger.info(f"Curves data saved to {curves_path}")
    
    # 5. 绘制对比图
    plot_path = os.path.join(output_dir, f'comparison_plot_{timestamp}.png')
    plot_comparison(evolution_curve, random_curve, evolution_all_ntk, random_all_ntk, plot_path)
    
    return stats, evolution_curve, random_curve, evolution_all_ntk, random_all_ntk


def main():
    parser = argparse.ArgumentParser(description='Compare Aging Evolution vs Random Search')
    parser.add_argument('--max_eval', type=int, default=100, 
                        help='Maximum number of evaluations for each algorithm')
    parser.add_argument('--pop_size', type=int, default=50,
                        help='Population size for Aging Evolution')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    run_experiment(
        max_evaluations=args.max_eval,
        population_size=args.pop_size,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
