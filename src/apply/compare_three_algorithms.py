# -*- coding: utf-8 -*-
"""
Compare three search algorithms and plot Pareto fronts.
"""


import os
import sys
import json
import time
import random
import argparse
import numpy as np
from collections import deque
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plotting import plot_pareto_comparison

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, FinalEvaluator, clear_gpu_memory
from models.network import NetworkBuilder
from utils.generation import generate_valid_child
from utils.logger import logger
from utils.constraints import update_param_bounds_for_dataset


class ModelInfo:
    def __init__(self, individual: Individual, param_count: float = 0, 
                 accuracy: float = 0, ntk_score: float = None):
        self.individual = individual
        self.param_count = param_count
        self.accuracy = accuracy
        self.ntk_score = ntk_score
        
    def to_dict(self):
        return {
            'id': self.individual.id,
            'encoding': self.individual.encoding,
            'param_count': self.param_count,
            'accuracy': self.accuracy,
            'ntk_score': self.ntk_score
        }


def count_parameters(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def get_model_param_count(individual: Individual) -> float:
    try:
        model = NetworkBuilder.build_from_individual(individual)
        return count_parameters(model)
    except Exception as e:
        logger.warning(f"Failed to build model for param count: {e}")
        return 0


class ThreeStageEA:
    
    def __init__(self, 
                 max_evaluations: int = 500,
                 population_size: int = 50,
                 top_n1: int = 20,
                 top_n2: int = 10,
                 short_epochs: int = 30,
                 full_epochs: int = 150):
        self.max_evaluations = max_evaluations
        self.population_size = population_size
        self.top_n1 = top_n1
        self.top_n2 = top_n2
        self.short_epochs = short_epochs
        self.full_epochs = full_epochs
        
        self.population = deque()
        self.history: List[Individual] = []
        self.final_models: List[ModelInfo] = []
        
    def _select_parents(self) -> Tuple[Individual, Individual]:
        current_pop_list = list(self.population)
        parents = selection_operator.tournament_selection(
            current_pop_list,
            tournament_size=min(config.TOURNAMENT_SIZE, len(current_pop_list)),
            num_winners=config.TOURNAMENT_WINNERS
        )
        if len(parents) < 2:
            return parents[0], parents[0]
        return parents[0], parents[1]
    
    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        def repair_fn(child: Individual) -> Individual:
            for _ in range(20):
                candidate = mutation_operator.mutate(random.choice([parent1, parent2]))
                if Encoder.validate_encoding(candidate.encoding):
                    return candidate
            return random.choice([parent1, parent2]).copy()

        return generate_valid_child(
            parent1=parent1,
            parent2=parent2,
            crossover_fn=crossover_operator.crossover,
            mutation_fn=mutation_operator.mutate,
            repair_fn=repair_fn,
            resample_fn=population_initializer.create_valid_individual,
            crossover_prob=config.PROB_CROSSOVER,
            mutation_prob=config.PROB_MUTATION,
            max_attempts=50,
        )
    
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        logger.info("=" * 60)
        logger.info("Three-Stage EA Algorithm Started")
        logger.info(f"Config: max_eval={self.max_evaluations}, pop={self.population_size}")
        logger.info(f"        top_n1={self.top_n1}, top_n2={self.top_n2}")
        logger.info("=" * 60)
        
        # ==================== Stage 1: NTK Screening ====================
        logger.info("\n[Stage 1] NTK Evaluation & Aging Evolution Search...")
        
        eval_count = 0
        
        while len(self.population) < self.population_size and eval_count < self.max_evaluations:
            ind = population_initializer.create_valid_individual()
            ind.id = eval_count
            fitness_evaluator.evaluate_individual(ind)
            self.population.append(ind)
            self.history.append(ind)
            eval_count += 1
            
            if eval_count % 20 == 0:
                logger.info(f"  Initialization progress: {eval_count}/{self.population_size}")
                clear_gpu_memory()
        
        while eval_count < self.max_evaluations:
            parent1, parent2 = self._select_parents()
            child = self._generate_offspring(parent1, parent2)
            child.id = eval_count
            
            fitness_evaluator.evaluate_individual(child)
            
            self.population.popleft()
            self.population.append(child)
            self.history.append(child)
            eval_count += 1
            
            if eval_count % 50 == 0:
                valid_fitnesses = [ind.fitness for ind in self.history 
                                   if ind.fitness is not None and ind.fitness < 100000]
                best_ntk = min(valid_fitnesses) if valid_fitnesses else float('inf')
                logger.info(f"  Evolution progress: {eval_count}/{self.max_evaluations}, Best NTK: {best_ntk:.2f}")
                clear_gpu_memory()
        
        logger.info(f"Stage 1 completed: Evaluated {len(self.history)} architectures")
        
        # ==================== Stage 2: Short Training Screening ====================
        logger.info(f"\n[Stage 2] Short Training Screening (Top {self.top_n1}, {self.short_epochs} epochs)...")
        
        unique_history = {}
        for ind in self.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            elif ind.fitness is not None:
                if unique_history[enc_tuple].fitness is None or ind.fitness < unique_history[enc_tuple].fitness:
                    unique_history[enc_tuple] = ind
        
        candidates = list(unique_history.values())
        candidates.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        top_n1_candidates = candidates[:self.top_n1]
        
        logger.info(f"  Selected Top {len(top_n1_candidates)} from {len(candidates)} unique architectures")
        
        if evaluator is None:
            evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
        
        short_train_results = []
        for i, ind in enumerate(top_n1_candidates):
            logger.info(f"  Short training [{i+1}/{len(top_n1_candidates)}] ID: {ind.id}")
            try:
                acc, _ = evaluator.evaluate_individual(ind, epochs=self.short_epochs)
                ind.quick_score = acc
                short_train_results.append(ind)
            except Exception as e:
                logger.warning(f"  Short training failed: {e}")
                ind.quick_score = 0
            clear_gpu_memory()
        
        # ==================== Stage 3: Full Training ====================
        logger.info(f"\n[Stage 3] Full Training (Top {self.top_n2}, {self.full_epochs} epochs)...")
        
        short_train_results.sort(key=lambda x: x.quick_score if x.quick_score else 0, reverse=True)
        top_n2_candidates = short_train_results[:self.top_n2]
        
        for i, ind in enumerate(top_n2_candidates):
            logger.info(f"  Full training [{i+1}/{len(top_n2_candidates)}] ID: {ind.id}")
            try:
                acc, _ = evaluator.evaluate_individual(ind, epochs=self.full_epochs)
                param_count = get_model_param_count(ind)
                
                model_info = ModelInfo(
                    individual=ind,
                    param_count=param_count,
                    accuracy=acc,
                    ntk_score=ind.fitness
                )
                self.final_models.append(model_info)
                logger.info(f"    Accuracy: {acc:.2f}%, Params: {param_count:.2f}M")
            except Exception as e:
                logger.warning(f"  Full training failed: {e}")
            clear_gpu_memory()
        
        logger.info(f"\nThree-Stage EA completed, obtained {len(self.final_models)} final models")
        return self.final_models


class TraditionalEA:
    
    def __init__(self, 
                 max_evaluations: int = 100,
                 population_size: int = 20,
                 top_n: int = 10,
                 search_epochs: int = 30,
                 full_epochs: int = 150):
        self.max_evaluations = max_evaluations
        self.population_size = population_size
        self.top_n = top_n
        self.search_epochs = search_epochs
        self.full_epochs = full_epochs
        
        self.population = deque()
        self.history: List[Individual] = []
        self.final_models: List[ModelInfo] = []
        
    def _select_parents(self) -> Tuple[Individual, Individual]:
        current_pop_list = list(self.population)
        tournament = random.sample(current_pop_list, min(config.TOURNAMENT_SIZE, len(current_pop_list)))
        tournament.sort(key=lambda x: x.fitness if x.fitness is not None else 0, reverse=True)
        
        if len(tournament) < 2:
            return tournament[0], tournament[0]
        return tournament[0], tournament[1]
    
    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        def repair_fn(child: Individual) -> Individual:
            for _ in range(20):
                candidate = mutation_operator.mutate(random.choice([parent1, parent2]))
                if Encoder.validate_encoding(candidate.encoding):
                    return candidate
            return random.choice([parent1, parent2]).copy()

        return generate_valid_child(
            parent1=parent1,
            parent2=parent2,
            crossover_fn=crossover_operator.crossover,
            mutation_fn=mutation_operator.mutate,
            repair_fn=repair_fn,
            resample_fn=population_initializer.create_valid_individual,
            crossover_prob=config.PROB_CROSSOVER,
            mutation_prob=config.PROB_MUTATION,
            max_attempts=50,
        )
    
    def _evaluate_fitness(self, ind: Individual, evaluator: FinalEvaluator) -> float:
        try:
            acc, _ = evaluator.evaluate_individual(ind, epochs=self.search_epochs)
            return acc
        except Exception as e:
            logger.warning(f"  Fitness evaluation failed: {e}")
            return 0.0
    
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        logger.info("=" * 60)
        logger.info("Traditional EA (Classic Aging Evolution) Started")
        logger.info(f"Config: max_eval={self.max_evaluations}, pop={self.population_size}")
        logger.info(f"        search_epochs={self.search_epochs}, full_epochs={self.full_epochs}")
        logger.info(f"        top_n={self.top_n}")
        logger.info("=" * 60)
        
        if evaluator is None:
            evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
        
        eval_count = 0
        best_fitness = 0.0
        
        # ==================== Stage 1: Aging Evolution Search ====================
        logger.info(f"\n[Stage 1] Aging Evolution Search (fitness={self.search_epochs} epoch accuracy)...")
        
        # 1.1 Initialize population
        logger.info(f"  Initializing population (size={self.population_size})...")
        while len(self.population) < self.population_size and eval_count < self.max_evaluations:
            ind = population_initializer.create_valid_individual()
            ind.id = eval_count
            
            fitness = self._evaluate_fitness(ind, evaluator)
            ind.fitness = fitness
            
            self.population.append(ind)
            self.history.append(ind)
            eval_count += 1
            
            best_fitness = max(best_fitness, fitness)
            
            if eval_count % 5 == 0:
                logger.info(f"    Init progress: {eval_count}/{self.population_size}, Best Acc: {best_fitness:.2f}%")
            clear_gpu_memory()
        
        logger.info(f"  Population initialized, current best: {best_fitness:.2f}%")
        
        # 1.2 Aging Evolution Loop
        logger.info(f"\n  Starting evolution loop...")
        while eval_count < self.max_evaluations:
            parent1, parent2 = self._select_parents()
            
            child = self._generate_offspring(parent1, parent2)
            child.id = eval_count
            
            fitness = self._evaluate_fitness(child, evaluator)
            child.fitness = fitness
            
            self.population.popleft()
            self.population.append(child)
            self.history.append(child)
            eval_count += 1
            
            best_fitness = max(best_fitness, fitness)
            
            if eval_count % 10 == 0:
                logger.info(f"    Evolution progress: {eval_count}/{self.max_evaluations}, Current: {fitness:.2f}%, Best: {best_fitness:.2f}%")
            clear_gpu_memory()
        
        logger.info(f"\n  Search completed, evaluated {len(self.history)} architectures, best: {best_fitness:.2f}%")
        
        # ==================== Stage 2: Full Training Top N ====================
        logger.info(f"\n[Stage 2] Full Training Top {self.top_n} Models ({self.full_epochs} epochs)...")
        
        unique_history = {}
        for ind in self.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            elif ind.fitness is not None:
                if unique_history[enc_tuple].fitness is None or ind.fitness > unique_history[enc_tuple].fitness:
                    unique_history[enc_tuple] = ind
        
        candidates = list(unique_history.values())
        candidates.sort(key=lambda x: x.fitness if x.fitness is not None else 0, reverse=True)
        top_candidates = candidates[:self.top_n]
        
        logger.info(f"  Selected Top {len(top_candidates)} from {len(candidates)} unique architectures")
        
        for i, ind in enumerate(top_candidates):
            logger.info(f"  Full training [{i+1}/{len(top_candidates)}] ID: {ind.id}, Search Acc: {ind.fitness:.2f}%")
            try:
                acc, _ = evaluator.evaluate_individual(ind, epochs=self.full_epochs)
                param_count = get_model_param_count(ind)
                
                model_info = ModelInfo(
                    individual=ind,
                    param_count=param_count,
                    accuracy=acc,
                    ntk_score=None
                )
                self.final_models.append(model_info)
                logger.info(f"    Full training accuracy: {acc:.2f}%, Params: {param_count:.2f}M")
            except Exception as e:
                logger.warning(f"  Full training failed: {e}")
            clear_gpu_memory()
        
        logger.info(f"\nTraditional EA completed, obtained {len(self.final_models)} final models")
        return self.final_models


class RandomSearchAlgorithm:
    
    def __init__(self, 
                 num_samples: int = 10,
                 full_epochs: int = 150):
        self.num_samples = num_samples
        self.full_epochs = full_epochs
        self.final_models: List[ModelInfo] = []
        
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        logger.info("=" * 60)
        logger.info("Random Search Started")
        logger.info(f"Config: num_samples={self.num_samples}, full_epochs={self.full_epochs}")
        logger.info("=" * 60)
        
        if evaluator is None:
            evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
        
        for i in range(self.num_samples):
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            logger.info(f"\n  Random sample [{i+1}/{self.num_samples}]")
            try:
                acc, _ = evaluator.evaluate_individual(ind, epochs=self.full_epochs)
                param_count = get_model_param_count(ind)
                
                model_info = ModelInfo(
                    individual=ind,
                    param_count=param_count,
                    accuracy=acc,
                    ntk_score=None
                )
                self.final_models.append(model_info)
                logger.info(f"    Accuracy: {acc:.2f}%, Params: {param_count:.2f}M")
            except Exception as e:
                logger.warning(f"  Training failed: {e}")
            clear_gpu_memory()
        
        logger.info(f"\nRandom Search completed, obtained {len(self.final_models)} final models")
        return self.final_models


def save_experiment_results(three_stage_models: List[ModelInfo],
                            traditional_models: List[ModelInfo],
                            random_models: List[ModelInfo],
                            output_dir: str,
                            config_dict: dict = None):
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'config': config_dict or {},
        'three_stage_ea': [m.to_dict() for m in three_stage_models],
        'traditional_ea': [m.to_dict() for m in traditional_models],
        'random_search': [m.to_dict() for m in random_models],
        'statistics': {
            'three_stage_ea': {
                'count': len(three_stage_models),
                'avg_accuracy': np.mean([m.accuracy for m in three_stage_models]) if three_stage_models else 0,
                'max_accuracy': max([m.accuracy for m in three_stage_models]) if three_stage_models else 0,
                'avg_params': np.mean([m.param_count for m in three_stage_models]) if three_stage_models else 0,
            },
            'traditional_ea': {
                'count': len(traditional_models),
                'avg_accuracy': np.mean([m.accuracy for m in traditional_models]) if traditional_models else 0,
                'max_accuracy': max([m.accuracy for m in traditional_models]) if traditional_models else 0,
                'avg_params': np.mean([m.param_count for m in traditional_models]) if traditional_models else 0,
            },
            'random_search': {
                'count': len(random_models),
                'avg_accuracy': np.mean([m.accuracy for m in random_models]) if random_models else 0,
                'max_accuracy': max([m.accuracy for m in random_models]) if random_models else 0,
                'avg_params': np.mean([m.param_count for m in random_models]) if random_models else 0,
            }
        }
    }
    
    output_path = os.path.join(output_dir, f'experiment_results_{timestamp}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path


def run_experiment(args):
    if args.dataset:
        config.FINAL_DATASET = args.dataset
    if args.optimizer:
        config.OPTIMIZER = args.optimizer
    update_param_bounds_for_dataset(config.FINAL_DATASET)
    
    logger.info("=" * 70)
    logger.info("         Three Algorithm Comparison Experiment")
    logger.info("=" * 70)
    logger.info(f"Experiment Config:")
    logger.info(f"  - Three-Stage EA: NTK evals={args.ts_ntk_evals}, short={args.ts_short_epochs}ep, full={args.full_epochs}ep")
    logger.info(f"  - Traditional EA: search evals={args.te_evals}, search={args.te_search_epochs}ep, Top{args.te_top_n} full={args.full_epochs}ep")
    logger.info(f"  - Random Search: samples={args.rs_samples}, full={args.full_epochs}ep")
    logger.info("=" * 70)
    
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'experiment_results'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
    
    three_stage_models = []
    traditional_models = []
    random_models = []
    
    # 1. Run Three-Stage EA
    if not args.skip_three_stage:
        logger.info("\n" + "=" * 50)
        logger.info("Starting: Three-Stage EA")
        logger.info("=" * 50)
        
        three_stage_ea = ThreeStageEA(
            max_evaluations=args.ts_ntk_evals,
            population_size=args.ts_pop_size,
            top_n1=args.ts_top_n1,
            top_n2=args.ts_top_n2,
            short_epochs=args.ts_short_epochs,
            full_epochs=args.full_epochs
        )
        three_stage_models = three_stage_ea.run(evaluator)
    
    # 2. Run Traditional EA
    if not args.skip_traditional:
        logger.info("\n" + "=" * 50)
        logger.info("Starting: Traditional EA (Classic Aging Evolution)")
        logger.info("=" * 50)
        
        traditional_ea = TraditionalEA(
            max_evaluations=args.te_evals,
            population_size=args.te_pop_size,
            top_n=args.te_top_n,
            search_epochs=args.te_search_epochs,
            full_epochs=args.full_epochs
        )
        traditional_models = traditional_ea.run(evaluator)
    
    # 3. Run Random Search
    if not args.skip_random:
        logger.info("\n" + "=" * 50)
        logger.info("Starting: Random Search")
        logger.info("=" * 50)
        
        random_search = RandomSearchAlgorithm(
            num_samples=args.rs_samples,
            full_epochs=args.full_epochs
        )
        random_models = random_search.run(evaluator)
    
    config_dict = vars(args)
    save_experiment_results(
        three_stage_models,
        traditional_models,
        random_models,
        output_dir,
        config_dict
    )
    
    plot_pareto_comparison(
        three_stage_models,
        traditional_models,
        random_models,
        output_dir,
        show_plot=not args.no_show
    )
    
    # Print final statistics
    logger.info("\n" + "=" * 70)
    logger.info("                    Experiment Results Summary")
    logger.info("=" * 70)
    
    def print_stats(name, models):
        if not models:
            logger.info(f"\n{name}: No results")
            return
        accs = [m.accuracy for m in models if m.accuracy > 0]
        params = [m.param_count for m in models if m.param_count > 0]
        if accs:
            logger.info(f"\n{name}:")
            logger.info(f"  Model count: {len(models)}")
            logger.info(f"  Accuracy range: {min(accs):.2f}% - {max(accs):.2f}%")
            logger.info(f"  Average accuracy: {np.mean(accs):.2f}%")
            logger.info(f"  Params range: {min(params):.2f}M - {max(params):.2f}M")
            logger.info(f"  Average params: {np.mean(params):.2f}M")
    
    print_stats("Three-Stage EA", three_stage_models)
    print_stats("Traditional EA", traditional_models)
    print_stats("Random Search", random_models)
    
    logger.info("\n" + "=" * 70)
    logger.info("Experiment completed!")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare three search algorithms")

    parser.add_argument(
        "--ts_ntk_evals",
        type=int,
        default=1000,
        help="Three-stage EA NTK evaluation count",
    )
    parser.add_argument(
        "--ts_pop_size",
        type=int,
        default=50,
        help="Three-stage EA population size",
    )
    parser.add_argument(
        "--ts_top_n1",
        type=int,
        default=20,
        help="Three-stage EA top N1 after NTK screening",
    )
    parser.add_argument(
        "--ts_top_n2",
        type=int,
        default=10,
        help="Three-stage EA top N2 after short training",
    )
    parser.add_argument(
        "--ts_short_epochs",
        type=int,
        default=30,
        help="Three-stage EA short training epochs",
    )

    parser.add_argument(
        "--te_evals",
        type=int,
        default=15,
        help="Traditional EA evaluation count",
    )
    parser.add_argument(
        "--te_pop_size",
        type=int,
        default=5,
        help="Traditional EA population size",
    )
    parser.add_argument(
        "--te_top_n",
        type=int,
        default=10,
        help="Traditional EA full-training top N",
    )
    parser.add_argument(
        "--te_search_epochs",
        type=int,
        default=30,
        help="Traditional EA search training epochs",
    )

    parser.add_argument(
        "--rs_samples",
        type=int,
        default=16,
        help="Random search sample count",
    )

    parser.add_argument(
        "--full_epochs",
        type=int,
        default=100,
        help="Full training epochs",
    )

    parser.add_argument(
        "--skip_three_stage",
        action="store_true",
        help="Skip three-stage EA",
    )
    parser.add_argument(
        "--skip_traditional",
        action="store_true",
        help="Skip traditional EA",
    )
    parser.add_argument(
        "--skip_random",
        action="store_true",
        help="Skip random search",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display plots",
    )

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode with reduced evaluations and epochs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cifar10", "cifar100"],
        help="Dataset to use (default: config.FINAL_DATASET)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=config.OPTIMIZER_OPTIONS,
        help="Optimizer to use (default: config.OPTIMIZER)",
    )

    args = parser.parse_args()

    if args.quick_test:
        args.ts_ntk_evals = 50
        args.ts_pop_size = 10
        args.ts_top_n1 = 5
        args.ts_top_n2 = 2
        args.ts_short_epochs = 1
        
        args.te_evals = 10
        args.te_pop_size = 5
        args.te_top_n = 2
        args.te_search_epochs = 1
        
        args.rs_samples = 4
        
        args.full_epochs = 1
        
        logger.info("Quick test mode enabled: Reduced parameters significantly")

    run_experiment(args)


if __name__ == "__main__":
    main()
