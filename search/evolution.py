# -*- coding: utf-8 -*-
"""
进化算法模块
实现三阶段进化流程
"""
import random
import os
import pickle
import time
from typing import List, Tuple, Optional
from utils.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator, adaptive_mutation_controller
from engine.evaluator import fitness_evaluator, FinalEvaluator
from search.nsga2 import NSGAII
from utils.logger import logger, tb_logger, failed_logger

class EvolutionaryNAS:
    def __init__(self, population_size: int = None, max_gen: int = None,
                 g1: int = None, g2: int = None):
        self.population_size = population_size or config.POPULATION_SIZE
        self.max_gen = max_gen or config.MAX_GEN
        self.g1 = g1 or config.G1
        self.g2 = g2 or config.G2
        
        self.population: List[Individual] = []
        self.current_gen = 0
        self.best_individual: Optional[Individual] = None
        self.history = []
        self.pareto_history = []
        
        self._log_search_space_info()

    def _log_search_space_info(self):
        logger.info(config.get_search_space_summary())
        logger.info(f"Pop Size: {self.population_size}, Max Gen: {self.max_gen}")

    def get_current_phase(self) -> int:
        if self.current_gen < self.g1: return 1
        elif self.current_gen < self.g2: return 2
        else: return 3

    def initialize_population(self):
        logger.info("Initializing population...")
        self.population = population_initializer.create_diverse_population(self.population_size)
        self.current_gen = 0
        self._validate_and_filter_population()
        self.population = self._deduplicate_population(self.population)
        if len(self.population) < self.population_size:
            self.population = self._fill_population(self.population, self.population_size)
        
        self._evaluate_population()
        self._update_best()
        logger.info(f"Population initialized with {len(self.population)} individuals")

    def _validate_and_filter_population(self):
        valid_population = []
        for ind in self.population:
            if Encoder.validate_encoding(ind.encoding):
                valid_population.append(ind)
            else:
                failed_logger.save_failed_individual(ind, "Invalid encoding", self.current_gen)
        
        while len(valid_population) < self.population_size:
            new_ind = population_initializer.create_valid_individual()
            if new_ind: valid_population.append(new_ind)
        self.population = valid_population

    def _deduplicate_population(self, population: List[Individual]) -> List[Individual]:
        seen_encodings = {}
        for ind in population:
            encoding_key = tuple(ind.encoding)
            if encoding_key not in seen_encodings:
                seen_encodings[encoding_key] = ind
            else:
                existing = seen_encodings[encoding_key]
                existing_fitness = existing.fitness if existing.fitness else float('-inf')
                new_fitness = ind.fitness if ind.fitness else float('-inf')
                if new_fitness > existing_fitness:
                    seen_encodings[encoding_key] = ind
        return list(seen_encodings.values())

    def _fill_population(self, population: List[Individual], target_size: int) -> List[Individual]:
        existing_encodings = {tuple(ind.encoding) for ind in population}
        attempts = 0
        while len(population) < target_size and attempts < target_size * 10:
            attempts += 1
            new_ind = population_initializer.create_valid_individual()
            if new_ind:
                new_encoding = tuple(new_ind.encoding)
                if new_encoding not in existing_encodings:
                    population.append(new_ind)
                    existing_encodings.add(new_encoding)
        return population

    def _evaluate_population(self):
        phase = self.get_current_phase()
        if phase == 1 or phase == 3:
            logger.info(f"Phase {phase}: Using NTK evaluation")
            fitness_evaluator.evaluate_population_ntk(self.population)
        else:
            logger.info("Phase 2: Using survival time and parameter count (NSGA-II)")
            fitness_evaluator.evaluate_population_survival(self.population, self.current_gen)

    def _generate_two_offspring(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() < config.PROB_CROSSOVER:
            child1, child2 = crossover_operator.crossover(parent1, parent2, self.current_gen)
            if random.random() < config.PROB_MUTATION: child1 = mutation_operator.mutate(child1, self.current_gen)
            if random.random() < config.PROB_MUTATION: child2 = mutation_operator.mutate(child2, self.current_gen)
        else:
            if random.random() < config.PROB_MUTATION:
                child1 = mutation_operator.mutate(parent1, self.current_gen)
            else:
                child1 = parent1.copy(); child1.birth_generation = self.current_gen
            
            if random.random() < config.PROB_MUTATION:
                child2 = mutation_operator.mutate(parent2, self.current_gen)
            else:
                child2 = parent2.copy(); child2.birth_generation = self.current_gen
                
        return self._try_repair(child1, [parent1, parent2]), self._try_repair(child2, [parent1, parent2])

    def _try_repair(self, ind: Individual, parents: List[Individual]) -> Individual:
        for _ in range(5):
            if Encoder.validate_encoding(ind.encoding): return ind
            parent = random.choice(parents)
            ind = mutation_operator.mutate(parent, self.current_gen)
        return random.choice(parents).copy()

    def _select_survivors(self, combined_population: List[Individual]) -> List[Individual]:
        phase = self.get_current_phase()
        if phase == 2:
            survivors = NSGAII.select_by_nsga2(combined_population, self.population_size)
            pareto_front = NSGAII.get_pareto_front(combined_population)
            self.pareto_history.append({
                'generation': self.current_gen,
                'pareto_front': [(ind.id, ind.survival_time, ind.param_count) for ind in pareto_front]
            })
            tb_logger.log_pareto_front(self.current_gen, pareto_front)
        else:
            sorted_pop = sorted(combined_population, key=lambda x: x.fitness if x.fitness else float('-inf'), reverse=True)
            survivors = sorted_pop[:self.population_size]
        return survivors

    def _update_best(self):
        if not self.population: return
        current_best = max(self.population, key=lambda x: x.fitness if x.fitness else float('-inf'))
        if (self.best_individual is None or 
            (current_best.fitness is not None and 
             (self.best_individual.fitness is None or current_best.fitness > self.best_individual.fitness))):
            self.best_individual = current_best.copy()
            logger.info(f"New best: ID={self.best_individual.id}, Fitness={self.best_individual.fitness}")
            logger.log_architecture(self.best_individual.id, self.best_individual.encoding, 
                                    self.best_individual.fitness, self.best_individual.param_count, "[NEW BEST] ")

    def _record_statistics(self):
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        if fitnesses:
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)
        else:
            avg_fitness = best_fitness = 0.0
            
        stats = {
            'generation': self.current_gen,
            'phase': self.get_current_phase(),
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
        }
        self.history.append(stats)
        logger.log_generation(self.current_gen, best_fitness, avg_fitness, len(self.population))
        tb_logger.log_generation_stats(self.current_gen, stats)
        
        if best_fitness > 0:
            adaptive_mutation_controller.update(self.current_gen, best_fitness, self.get_current_phase())

    def evolve_one_generation(self):
        self.current_gen += 1
        logger.log_phase_change(self.current_gen, self.get_current_phase())
        
        offspring_list = []
        num_pairs = self.population_size // 2
        for _ in range(num_pairs):
            parents = selection_operator.tournament_selection(self.population)
            if len(parents) < 2: parents = parents * 2
            c1, c2 = self._generate_two_offspring(parents[0], parents[1])
            offspring_list.extend([c1, c2])
            
        if self.population_size % 2 != 0:
             parents = selection_operator.tournament_selection(self.population)
             c1, _ = self._generate_two_offspring(parents[0], parents[0])
             offspring_list.append(c1)

        # Evaluate offspring
        phase = self.get_current_phase()
        if phase in [1, 3]:
            fitness_evaluator.evaluate_population_ntk(offspring_list)
        else:
            for ind in offspring_list: ind.survival_time = 0
            fitness_evaluator.evaluate_population_survival(offspring_list, self.current_gen)

        combined = self.population + offspring_list
        combined = self._deduplicate_population(combined)
        self.population = self._select_survivors(combined)
        
        if len(self.population) < self.population_size:
            self.population = self._fill_population(self.population, self.population_size)
            
        self._update_best()
        self._record_statistics()

    def run(self, num_generations: int = None, final_eval: bool = True):
        if num_generations is None: num_generations = self.max_gen
        logger.info(f"Starting evolution for {num_generations} generations")
        
        start_time = time.time()
        if not self.population:
            self.initialize_population()
            self._record_statistics()
            
        for _ in range(num_generations):
            self.evolve_one_generation()
            if config.SAVE_CHECKPOINT and self.current_gen % 10 == 0:
                self.save_checkpoint()
                
        tb_logger.close()
        logger.info(f"Evolution completed in {time.time() - start_time:.2f}s")
        
        if final_eval:
            return self.final_evaluation()
        return self.best_individual

    def final_evaluation(self):
        evaluator = FinalEvaluator()
        return evaluator.evaluate_top_individuals(self.population)

    def save_checkpoint(self, filepath: str = None):
        if filepath is None:
            if not os.path.exists(config.CHECKPOINT_DIR): os.makedirs(config.CHECKPOINT_DIR)
            filepath = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_gen{self.current_gen}.pkl')
        
        checkpoint = {
            'current_gen': self.current_gen,
            'population': self.population,
            'best_individual': self.best_individual,
            'history': self.history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.current_gen = checkpoint['current_gen']
        self.population = checkpoint['population']
        self.best_individual = checkpoint['best_individual']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {filepath}")
