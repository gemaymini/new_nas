# -*- coding: utf-8 -*-
"""
Compare aging evolution vs random search.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, clear_gpu_memory
from utils.logger import logger


class RandomSearch:
    
    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations
        self.history: List[Individual] = []
        self.best_fitness_curve: List[float] = []
        self.all_fitness_values: List[float] = []
        
    def run(self) -> Tuple[List[Individual], List[float], List[float]]:
        logger.info(f"Starting Random Search for {self.max_evaluations} evaluations...")
        
        best_fitness = float('inf')
        
        for i in range(self.max_evaluations):
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            fitness_evaluator.evaluate_individual(ind)
            self.history.append(ind)
            
            current_fitness = ind.fitness if ind.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
            if ind.fitness is not None and ind.fitness < 100000:
                best_fitness = min(best_fitness, ind.fitness)
            
            self.best_fitness_curve.append(best_fitness if best_fitness < float('inf') else 100000)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Random Search [{i+1}/{self.max_evaluations}] Best NTK: {best_fitness:.4f}")
                clear_gpu_memory()
        
        logger.info(f"Random Search completed. Best NTK: {best_fitness:.4f}")
        return self.history, self.best_fitness_curve, self.all_fitness_values


class AgingEvolutionSearch:
    
    def __init__(self, max_evaluations: int, population_size: int = None):
        self.max_evaluations = max_evaluations
        self.population_size = population_size or config.POPULATION_SIZE
        self.population = deque()
        self.history: List[Individual] = []
        self.best_fitness_curve: List[float] = []
        self.all_fitness_values: List[float] = []
        
    def _select_parents(self) -> Tuple[Individual, Individual]:
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
        child = None
        
        if random.random() < config.PROB_CROSSOVER:
            c1, c2, _ = crossover_operator.crossover(parent1, parent2)
            child = random.choice([c1, c2])
        else:
            child = random.choice([parent1, parent2]).copy()
        
        if random.random() < config.PROB_MUTATION:
            child = mutation_operator.mutate(child)
        
        if not Encoder.validate_encoding(child.encoding):
            for _ in range(20):
                child = mutation_operator.mutate(random.choice([parent1, parent2]))
                if Encoder.validate_encoding(child.encoding):
                    break
            else:
                child = random.choice([parent1, parent2]).copy()
        
        return child
    
    def run(self) -> Tuple[List[Individual], List[float], List[float]]:
        logger.info(f"Starting Aging Evolution for {self.max_evaluations} evaluations...")
        logger.info(f"Population size: {self.population_size}")
        
        best_fitness = float('inf')
        eval_count = 0
        
        logger.info("Initializing population...")
        while len(self.population) < self.population_size and eval_count < self.max_evaluations:
            ind = population_initializer.create_valid_individual()
            ind.id = eval_count
            fitness_evaluator.evaluate_individual(ind)
            
            self.population.append(ind)
            self.history.append(ind)
            
            current_fitness = ind.fitness if ind.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
            if ind.fitness is not None and ind.fitness < 100000:
                best_fitness = min(best_fitness, ind.fitness)
            
            self.best_fitness_curve.append(best_fitness if best_fitness < float('inf') else 100000)
            eval_count += 1
            
            if eval_count % 10 == 0:
                logger.info(f"Initialization [{eval_count}/{self.population_size}]")
        
        logger.info(f"Population initialized. Size: {len(self.population)}")
        
        while eval_count < self.max_evaluations:
            parent1, parent2 = self._select_parents()
            
            child = self._generate_offspring(parent1, parent2)
            child.id = eval_count
            
            fitness_evaluator.evaluate_individual(child)
            
            self.population.popleft()
            self.population.append(child)
            self.history.append(child)
            
            current_fitness = child.fitness if child.fitness is not None else 100000
            self.all_fitness_values.append(current_fitness)
            
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
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    
    if title is None:
        title = f'Aging Evolution vs Random Search (N={len(evolution_curve)})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    steps = list(range(1, len(evolution_curve) + 1))
    
    evo_valid_vals = [v for v in evolution_all_ntk if v < 100000]
    rand_valid_vals = [v for v in random_all_ntk if v < 100000]
    
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
    
    ax3 = axes[1, 0]
    color1, color2 = 'blue', 'red'
    
    line1, = ax3.plot(steps, evolution_curve, color=color1, linewidth=2, label='Aging Evolution')
    ax3.set_xlabel('Evaluation Count')
    ax3.set_ylabel('Aging Evolution Best NTK', color=color1)
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.set_ylim(min(evolution_curve) * 0.9, max(evolution_curve) * 1.1)
    
    ax3_twin = ax3.twinx()
    line2, = ax3_twin.plot(steps, random_curve, color=color2, linewidth=2, label='Random Search')
    ax3_twin.set_ylabel('Random Search Best NTK', color=color2)
    ax3_twin.tick_params(axis='y', labelcolor=color2)
    ax3_twin.set_ylim(min(random_curve) * 0.9, max(random_curve) * 1.1)
    
    ax3.set_title('3. Cumulative Best NTK (Dual Y-Axis)')
    ax3.legend([line1, line2], ['Aging Evolution', 'Random Search'], loc='upper right')
    ax3.grid(True, alpha=0.3)
    
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
    
    ax5 = axes[2, 0]
    evo_improvement = [(evolution_curve[0] - v) / evolution_curve[0] * 100 for v in evolution_curve]
    rand_improvement = [(random_curve[0] - v) / random_curve[0] * 100 for v in random_curve]
    ax5.plot(steps, evo_improvement, 'b-', linewidth=2, label='Aging Evolution')
    ax5.plot(steps, rand_improvement, 'r-', linewidth=2, label='Random Search')
    ax5.set_xlabel('Evaluation Count')
    ax5.set_ylabel('Improvement from Initial (%)')
    ax5.set_title('5. Relative Improvement Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.annotate(f'{evo_improvement[-1]:.1f}%', xy=(len(steps), evo_improvement[-1]), 
                 xytext=(5, 0), textcoords='offset points', fontsize=9, color='blue')
    ax5.annotate(f'{rand_improvement[-1]:.1f}%', xy=(len(steps), rand_improvement[-1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
    
    ax6 = axes[2, 1]
    window_size = max(5, len(evolution_all_ntk) // 20)
    
    def moving_average_continuous(data, window):
        cleaned = np.array([v if v < 100000 else np.nan for v in data], dtype=float)
        
        valid_mask = ~np.isnan(cleaned)
        if np.sum(valid_mask) >= 2:
            indices = np.arange(len(cleaned))
            cleaned = np.interp(indices, indices[valid_mask], cleaned[valid_mask])
        
        result = np.convolve(cleaned, np.ones(window)/window, mode='valid')
        padding = [np.mean(cleaned[:i+1]) for i in range(window-1)]
        return padding + list(result)
    
    evo_ma = moving_average_continuous(evolution_all_ntk, window_size)
    rand_ma = moving_average_continuous(random_all_ntk, window_size)
    
    ax6.semilogy(steps, evo_ma, 'b-', linewidth=2, label=f'Aging Evolution (MA={window_size})')
    ax6.semilogy(steps, rand_ma, 'r-', linewidth=2, label=f'Random Search (MA={window_size})')
    ax6.set_xlabel('Evaluation Count')
    ax6.set_ylabel('Moving Average NTK (log scale)')
    ax6.set_title('6. Moving Average NTK')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')
    
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
    
    ax8 = axes[3, 1]
    ax8.axis('off')
    
    evo_valid = [v for v in evolution_all_ntk if v < 100000]
    rand_valid = [v for v in random_all_ntk if v < 100000]
    
    table_data = [
        ['Metric', 'Aging Evolution', 'Random Search'],
        ['Best NTK', f'{min(evo_valid):.2f}' if evo_valid else 'N/A', 
         f'{min(rand_valid):.2f}' if rand_valid else 'N/A'],
        ['Mean NTK', f'{np.mean(evo_valid):.2f}' if evo_valid else 'N/A',
         f'{np.mean(rand_valid):.2f}' if rand_valid else 'N/A'],
        ['Std NTK', f'{np.std(evo_valid):.2f}' if evo_valid else 'N/A',
         f'{np.std(rand_valid):.2f}' if rand_valid else 'N/A'],
        ['Median NTK', f'{np.median(evo_valid):.2f}' if evo_valid else 'N/A',
         f'{np.median(rand_valid):.2f}' if rand_valid else 'N/A'],
        ['Valid Count', f'{len(evo_valid)}/{len(evolution_all_ntk)}',
         f'{len(rand_valid)}/{len(random_all_ntk)}'],
        ['Final Best', f'{evolution_curve[-1]:.2f}', f'{random_curve[-1]:.2f}'],
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
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plot saved to {output_path}")


def compute_statistics(evolution_history: List[Individual], random_history: List[Individual],
                       evolution_curve: List[float], random_curve: List[float]) -> dict:
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
    
    if stats['evolution']['best_ntk'] and stats['random']['best_ntk']:
        improvement = ((stats['random']['best_ntk'] - stats['evolution']['best_ntk']) 
                       / stats['random']['best_ntk'] * 100)
        stats['comparison']['best_ntk_improvement_%'] = round(float(improvement), 2)
        stats['comparison']['evolution_better'] = bool(stats['evolution']['best_ntk'] < stats['random']['best_ntk'])
    
    if evolution_curve and random_curve:
        threshold = random_curve[-1] * 1.05
        
        evo_converge_step = None
        for i, v in enumerate(evolution_curve):
            if v <= threshold:
                evo_converge_step = i + 1
                break
        
        rand_converge_step = len(random_curve)
        
        stats['comparison']['threshold'] = float(threshold)
        stats['comparison']['evolution_steps_to_threshold'] = evo_converge_step
        stats['comparison']['random_steps_to_threshold'] = rand_converge_step
        
        if evo_converge_step:
            speedup = rand_converge_step / evo_converge_step
            stats['comparison']['speedup'] = round(float(speedup), 2)
    
    return stats


def run_experiment(max_evaluations: int, population_size: int, seed: int = None, 
                   output_dir: str = None):
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
    
    logger.info("\n" + "=" * 40)
    logger.info("Phase 1: Running Aging Evolution")
    logger.info("=" * 40)
    start_time = time.time()
    evolution_search = AgingEvolutionSearch(max_evaluations, population_size)
    evolution_history, evolution_curve, evolution_all_ntk = evolution_search.run()
    evolution_time = time.time() - start_time
    logger.info(f"Aging Evolution time: {evolution_time:.2f}s")
    
    clear_gpu_memory()
    
    logger.info("\n" + "=" * 40)
    logger.info("Phase 2: Running Random Search")
    logger.info("=" * 40)
    start_time = time.time()
    random_search = RandomSearch(max_evaluations)
    random_history, random_curve, random_all_ntk = random_search.run()
    random_time = time.time() - start_time
    logger.info(f"Random Search time: {random_time:.2f}s")
    
    clear_gpu_memory()
    
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
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    stats_path = os.path.join(output_dir, f'comparison_stats_{timestamp}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistics saved to {stats_path}")
    
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
    
    plot_path = os.path.join(output_dir, f'comparison_plot_{timestamp}.png')
    plot_comparison(evolution_curve, random_curve, evolution_all_ntk, random_all_ntk, plot_path)
    
    return stats, evolution_curve, random_curve, evolution_all_ntk, random_all_ntk


def main():
    parser = argparse.ArgumentParser(description='Compare Aging Evolution vs Random Search')
    parser.add_argument('--max_eval', type=int, default=10, 
                        help='Maximum number of evaluations for each algorithm')
    parser.add_argument('--pop_size', type=int, default=5,
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
