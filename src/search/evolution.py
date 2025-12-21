# -*- coding: utf-8 -*-
"""
Aging Evolution (Regularized Evolution) Algorithm Implementation
"""
import random
import os
import pickle
import time
import threading
from collections import deque
from typing import List, Tuple, Optional
from copy import deepcopy

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, FinalEvaluator
from utils.logger import logger, tb_logger, failed_logger

class AgingEvolutionNAS:
    def __init__(self):
        self.population_size = config.POPULATION_SIZE
        self.max_gen = config.MAX_GEN # Total individuals to evaluate
        
        # 1. Population & History Management
        # Using deque for FIFO queue (fixed size handled by manual popleft)
        self.population = deque() 
        self.history: List[Individual] = []
        self.lock = threading.Lock()
        
        self.start_time = time.time()
        
        self._log_search_space_info()

    def _log_search_space_info(self):
        logger.info(config.get_search_space_summary())
        logger.info(f"Aging Evolution Config: Pop Size={self.population_size}, Total Gen={self.max_gen}")

    def initialize_population(self):
        """
        Initialize the population with random individuals until queue is full.
        """
        logger.info("Initializing population...")
        
        while len(self.population) < self.population_size:
            ind = population_initializer.create_valid_individual()
            # Evaluate immediately
            ind.id=len(self.population)
            fitness_evaluator.evaluate_individual(ind)
            self.population.append(ind)
            self.history.append(ind)
            
            if len(self.population) % 10 == 0:
                logger.info(f"Initialized {len(self.population)}/{self.population_size} individuals")

        logger.info(f"Population initialized. Size: {len(self.population)}")
        self._record_statistics()

    def _select_parents(self) -> Tuple[Individual, Individual]:
        """
        Tournament selection to choose 2 parents.
        """
        # Convert deque to list for random sampling
        current_pop_list = list(self.population)
        
        # Tournament selection returns sorted winners (best first)
        parents = selection_operator.tournament_selection(
            current_pop_list, 
            tournament_size=config.TOURNAMENT_SIZE,
            num_winners=config.TOURNAMENT_WINNERS
        )
        
        # If not enough parents (shouldn't happen if pop_size >= 2), duplicate best
        if len(parents) < 2:
            return parents[0], parents[0]
            
        return parents[0], parents[1]

    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Generate ONE offspring using Crossover and Mutation.
        """
        child = None
        
        # Crossover
        if random.random() < config.PROB_CROSSOVER:
            # Generate 2 children, pick random one
            c1, c2 = crossover_operator.crossover(parent1, parent2)
            child = random.choice([c1, c2])
        else:
            child = random.choice([parent1, parent2]).copy()

        # Mutation
        if random.random() < config.PROB_MUTATION:
            child = mutation_operator.mutate(child)
            
        # Validate and Repair
        if not Encoder.validate_encoding(child.encoding):
            child = self._repair_individual(child, [parent1, parent2])
            
        return child

    def _repair_individual(self, ind: Individual, parents: List[Individual]) -> Individual:
        for _ in range(20):
            ind = mutation_operator.mutate(random.choice(parents))
            if Encoder.validate_encoding(ind.encoding): return ind
        return random.choice(parents).copy()

    def step(self):
        """
        Perform one step of Aging Evolution:
        1. Select parents
        2. Generate child
        3. Evaluate child
        4. Atomic update: Push child, Pop oldest
        """
        
        # 1. Select Parents
        parent1, parent2 = self._select_parents()
        
        # 2. Generate Offspring
        child = self._generate_offspring(parent1, parent2)
        child.id = len(self.history)  # Assign new ID based on total history
        
        # 3. Evaluate (Calculate NTK)
        fitness_evaluator.evaluate_individual(child)
        
        # 4. Atomic Update
        with self.lock:
            # Remove oldest (head of deque)
            removed_ind = self.population.popleft()
            
            # Add new (tail of deque)
            self.population.append(child)
            
            # Add to history
            self.history.append(child)
            
        # Logging
        if len(self.history) % 10 == 0:
            logger.info(f"Step {len(self.history)-len(self.population)}/{self.max_gen}: Child Fitness={child.fitness:.4f}")
            self._record_statistics()

    def run_search(self):
        """
        Main loop for Aging Evolution Search.
        """
        logger.info(f"Starting Aging Evolution Search for {self.max_gen} steps...")
        
        if not self.population:
            self.initialize_population()
            
        # Continue until we have generated MAX_GEN individuals (including initial pop)
        # Or just run MAX_GEN steps? Usually MAX_GEN implies total evaluations.
        # Let's say we run until len(history) >= MAX_GEN
        
        while len(self.history) - len(self.population) < self.max_gen:
            self.step()
            
            if (len(self.history) - len(self.population)) % 100 == 0:
                self.save_checkpoint()

        logger.info("Search completed.")
        self.save_checkpoint()

    def run_screening_and_training(self):
        """
        Multi-stage screening and final training.
        """
        logger.info("Starting Screening and Training Phase...")
        
        # 1. History Screening (Top N1 by NTK)
        # Deduplicate history first based on encoding
        unique_history = {}
        for ind in self.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            else:
                # fitness 越小越好，保留更小的
                if ind.fitness is not None and (unique_history[enc_tuple].fitness is None or ind.fitness < unique_history[enc_tuple].fitness):
                    unique_history[enc_tuple] = ind
        
        candidates = list(unique_history.values())
        # fitness 越小越好，升序排列
        candidates.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'), reverse=False)
        
        top_n1 = candidates[:config.HISTORY_TOP_N1]
        logger.info(f"Selected Top {config.HISTORY_TOP_N1} candidates from {len(candidates)} unique history individuals based on NTK.")
        
        # 2. Short Training (Top N1 -> Val Acc)
        logger.info(f"Starting Short Training ({config.SHORT_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N1}...")
        
        # We use FinalEvaluator but with fewer epochs
        # Note: FinalEvaluator usually saves models. We might want to disable saving for short train or overwrite.
        # Let's use a temporary evaluator or just FinalEvaluator.
        
        evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
        
        short_results = []
        for i, ind in enumerate(top_n1):
            logger.info(f"Short Train [{i+1}/{len(top_n1)}] ID: {ind.id}")
            # Use 'quick_score' to store val acc from short training to avoid overwriting 'fitness' (NTK)
            # But FinalEvaluator returns best_acc.
            acc, _ = evaluator.evaluate_individual(ind, epochs=config.SHORT_TRAIN_EPOCHS)
            ind.quick_score = acc # Store for sorting
            short_results.append(ind)
            
        # 3. Select Top N2 (by Val Acc)
        short_results.sort(key=lambda x: x.quick_score if x.quick_score else float('-inf'), reverse=True)
        top_n2 = short_results[:config.HISTORY_TOP_N2]
        logger.info(f"Selected Top {config.HISTORY_TOP_N2} candidates based on Short Training Accuracy.")
        
        # 4. Full Training (Top N2 -> Final Model)
        logger.info(f"Starting Full Training ({config.FULL_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N2}...")
        
        final_results = []
        best_final_ind = None
        best_final_acc = 0.0
        
        for i, ind in enumerate(top_n2):
            logger.info(f"Full Train [{i+1}/{len(top_n2)}] ID: {ind.id}")
            acc, result = evaluator.evaluate_individual(ind, epochs=config.FULL_TRAIN_EPOCHS)
            
            # Log result
            logger.info(f"Individual {ind.id} Final Accuracy: {acc:.2f}%")
            
            if acc > best_final_acc:
                best_final_acc = acc
                best_final_ind = ind
                
            final_results.append(result)
            
        logger.info(f"Best Final Model: ID={best_final_ind.id}, Acc={best_final_acc:.2f}%")
        return best_final_ind

    def _record_statistics(self):
        # 排除极差的 fitness（100000）来计算平均值
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        valid_fitnesses = [f for f in fitnesses if f < 100000.0]
        
        if valid_fitnesses:
            avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
            best_fitness = min(valid_fitnesses)  # fitness 越小越好
        elif fitnesses:
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = min(fitnesses)
        else:
            avg_fitness = best_fitness = 0.0
            
        stats = {
            'generation': len(self.history)-len(self.population),
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'population_size': len(self.population)
        }
        
        # Use existing logger methods (might need adaptation)
        logger.log_generation(len(self.history)-len(self.population), best_fitness, avg_fitness, len(self.population))
        tb_logger.log_generation_stats(len(self.history)-len(self.population), stats)
        
        # Unit stats
        unit_counts = {}
        for ind in self.population:
            if ind.encoding:
                unit_num = ind.encoding[0]
                unit_counts[unit_num] = unit_counts.get(unit_num, 0) + 1
        logger.log_unit_stats(len(self.history)-len(self.population), unit_counts)

    def save_checkpoint(self, filepath: str = None):
        if filepath is None:
            if not os.path.exists(config.CHECKPOINT_DIR): os.makedirs(config.CHECKPOINT_DIR)
            filepath = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_step{len(self.history)-len(self.population)}.pkl')
        
        checkpoint = {
            'population': list(self.population), # Convert deque to list for pickling
            'history': self.history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.population = deque(checkpoint['population'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {filepath}")

