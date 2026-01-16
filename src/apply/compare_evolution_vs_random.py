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
# from search.mutation import mutation_operator, selection_operator, crossover_operator # No longer needed here
from engine.evaluator import fitness_evaluator, clear_gpu_memory
# from utils.generation import generate_valid_child # No longer needed here
from utils.logger import logger
from utils.constraints import update_param_bounds_for_dataset
from utils.plotting import plot_comparison


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



from search.evolution import AgingEvolutionNAS

def run_evolution_phase(max_evaluations: int, population_size: int) -> Tuple[List[Individual], List[float], List[float]]:
    """
    Run the standard AgingEvolutionNAS and extract statistics for comparison.
    """
    # Configure global settings for the run
    config.MAX_GEN = max_evaluations
    config.POPULATION_SIZE = population_size
    
    # Initialize and run
    nas = AgingEvolutionNAS()
    nas.run_search()
    
    # Extract data for plotting
    history = nas.history
    
    # Reconstruct curves from ntk_history (format: step, id, ntk, encoding)
    # Filter valid scores
    all_fitness_values = []
    best_fitness_curve = []
    
    current_best = float('inf')
    
    # We need to reconstruct the sequential list of fitness values to match the plotting format
    
    for entry in nas.ntk_history:
        fitness = entry[2]
        if fitness is None: 
            fitness = 100000.0
            
        all_fitness_values.append(fitness)
        
        if fitness < 100000:
            if fitness < current_best:
                current_best = fitness
        
        best_fitness_curve.append(current_best if current_best < float('inf') else 100000.0)
            
    return history, best_fitness_curve, all_fitness_values




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
                   output_dir: str = None, dataset: str = None):
    if dataset:
        config.FINAL_DATASET = dataset
    update_param_bounds_for_dataset(config.FINAL_DATASET)

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
    logger.info("=" * 40)
    start_time = time.time()
    
    # Use the refactored evolution phase runner
    evolution_history, evolution_curve, evolution_all_ntk = run_evolution_phase(max_evaluations, population_size)
    
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
    parser.add_argument('--dataset', type=str, default=None, choices=['cifar10', 'cifar100'],
                        help='Dataset to use for NTK bounds (default: config.FINAL_DATASET)')
    
    args = parser.parse_args()
    
    run_experiment(
        max_evaluations=args.max_eval,
        population_size=args.pop_size,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )


if __name__ == '__main__':
    main()
