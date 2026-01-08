# -*- coding: utf-8 -*-
"""
Aging Evolution (Regularized Evolution) Algorithm Implementation
"""
import random
import os
import pickle
import time
import threading
import json
import matplotlib.pyplot as plt
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
        
        # 去重机制：编码哈希 -> Individual 映射
        # 用于快速查找已评估的编码，避免重复评估
        self.encoding_cache: dict = {}  # encoding_key -> Individual
        self.duplicate_count = 0  # 记录跳过的重复个体数量
        
        # 搜索历史记录，用于分析和绘制曲线
        # 格式: [(step, individual_id, ntk_value, fitness, param_count, encoding), ...]
        self.ntk_history: List[Tuple[int, int, float, float, int, list]] = []
        
        self.start_time = time.time()
        
        # 时间记录
        self.search_time = 0.0  # 搜索阶段时间（秒）
        self.short_train_time = 0.0  # 短轮次训练时间（秒）
        self.full_train_time = 0.0  # 完整训练时间（秒）
        self.time_stats: dict = {}  # 详细时间统计
        
        self._log_search_space_info()
    
    @staticmethod
    def _encoding_to_key(encoding: List[int]) -> tuple:
        """将编码转换为可哈希的key用于去重"""
        return tuple(encoding)
    
    def _register_individual(self, ind: Individual):
        """注册个体到缓存中"""
        key = self._encoding_to_key(ind.encoding)
        if key not in self.encoding_cache:
            self.encoding_cache[key] = ind
    
    def _find_duplicate(self, encoding: List[int]) -> Optional[Individual]:
        """查找是否存在相同编码的已评估个体"""
        key = self._encoding_to_key(encoding)
        return self.encoding_cache.get(key, None)

    def _log_search_space_info(self):
        logger.info(config.get_search_space_summary())
        logger.info(f"Aging Evolution Config: Pop Size={self.population_size}, Total Gen={self.max_gen}")

    def initialize_population(self):
        """
        Initialize the population with random individuals until queue is full.
        去重：确保初始种群中没有重复编码
        """
        logger.info("Initializing population...")
        
        while len(self.population) < self.population_size:
            ind = population_initializer.create_valid_individual()
            
            # 检查是否已存在相同编码
            existing = self._find_duplicate(ind.encoding)
            if existing is not None:
                # 复用已有评估结果
                ind.fitness = existing.fitness
                ind.ntk_score = existing.ntk_score
                ind.param_count = existing.param_count
                self.duplicate_count += 1
            else:
                # 新编码，需要评估
                ind.id = len(self.history)
                fitness_evaluator.evaluate_individual(ind)
                self._register_individual(ind)
            
            self.population.append(ind)
            self.history.append(ind)
            
            # 记录评估结果
            step = 0  # 初始化阶段step=0
            self.ntk_history.append((step, ind.id, getattr(ind, 'ntk_score', None), ind.fitness, ind.param_count, ind.encoding.copy()))
            
            if len(self.population) % 10 == 0:
                logger.info(f"Initialized {len(self.population)}/{self.population_size} individuals (duplicates skipped: {self.duplicate_count})")

        logger.info(f"Population initialized. Size: {len(self.population)}, Unique: {len(self.encoding_cache)}")
        self._record_statistics()
        
        # 保存NTK曲线
        self._save_ntk_history()

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
        for _ in range(1000):
            ind = mutation_operator.mutate(random.choice(parents))
            if Encoder.validate_encoding(ind.encoding): return ind
        return random.choice(parents).copy()

    def step(self):
        """
        Perform one step of Aging Evolution:
        1. Select parents
        2. Generate child (with deduplication)
        3. Evaluate child (skip if duplicate)
        4. Atomic update: Push child, Pop oldest
        """
        
        # 1. Select Parents
        parent1, parent2 = self._select_parents()
        
        # 2. Generate Offspring (尝试生成不重复的子代)
        max_attempts = 10  # 最多尝试次数
        child = None
        is_duplicate = False
        
        for attempt in range(max_attempts):
            candidate = self._generate_offspring(parent1, parent2)
            existing = self._find_duplicate(candidate.encoding)
            
            if existing is None:
                # 新编码，使用它
                child = candidate
                break
            elif attempt == max_attempts - 1:
                # 最后一次尝试仍然重复，接受重复但复用结果
                child = candidate
                child.fitness = existing.fitness
                child.ntk_score = existing.ntk_score
                child.param_count = existing.param_count
                is_duplicate = True
                self.duplicate_count += 1
        
        child.id = len(self.history)  # Assign new ID based on total history
        
        # 3. Evaluate (Calculate NTK) - 仅对新编码评估
        if not is_duplicate:
            fitness_evaluator.evaluate_individual(child)
            self._register_individual(child)
        
        # 当前step（进化代数）
        current_step = len(self.history) - len(self.population) + 1
        
        # 记录评估结果
        self.ntk_history.append((current_step, child.id, getattr(child, 'ntk_score', None), child.fitness, child.param_count, child.encoding.copy()))
        
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
            # 每10步保存一次NTK历史
            self._save_ntk_history()

    def run_search(self):
        """
        Main loop for Aging Evolution Search.
        """
        logger.info(f"Starting Aging Evolution Search for {self.max_gen} steps...")
        search_start_time = time.time()
        
        if not self.population:
            self.initialize_population()
            
        # Continue until we have generated MAX_GEN individuals (including initial pop)
        # Or just run MAX_GEN steps? Usually MAX_GEN implies total evaluations.
        # Let's say we run until len(history) >= MAX_GEN
        
        while len(self.history) - len(self.population) < self.max_gen:
            self.step()
            
            if (len(self.history) - len(self.population)) % 100 == 0:
                self.save_checkpoint()

        # 记录搜索阶段时间
        self.search_time = time.time() - search_start_time
        logger.info(f"Search completed. Search time: {self._format_time(self.search_time)}")
        logger.info(f"Deduplication stats: Total={len(self.history)}, Unique={len(self.encoding_cache)}, Duplicates skipped={self.duplicate_count}")
        self.save_checkpoint()
        
        # 搜索结束后保存历史并绘制可视化图
        self._save_ntk_history()
        self.plot_ntk_curve()
        self.plot_fitness_curve()
        self.plot_param_fitness_correlation()

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
        short_train_start_time = time.time()
        
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
        
        # 记录短轮次训练时间
        self.short_train_time = time.time() - short_train_start_time
        logger.info(f"Short Training completed. Time: {self._format_time(self.short_train_time)}")
            
        # 3. Select Top N2 (by Val Acc)
        short_results.sort(key=lambda x: x.quick_score if x.quick_score else float('-inf'), reverse=True)
        top_n2 = short_results[:config.HISTORY_TOP_N2]
        logger.info(f"Selected Top {config.HISTORY_TOP_N2} candidates based on Short Training Accuracy.")
        
        # 4. Full Training (Top N2 -> Final Model)
        logger.info(f"Starting Full Training ({config.FULL_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N2}...")
        full_train_start_time = time.time()
        
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
        
        # 记录完整训练时间
        self.full_train_time = time.time() - full_train_start_time
        logger.info(f"Full Training completed. Time: {self._format_time(self.full_train_time)}")
        
        # 保存时间统计并打印总结
        self._save_time_stats()
            
        logger.info(f"Best Final Model: ID={best_final_ind.id}, Acc={best_final_acc:.2f}%")
        return best_final_ind

    def _record_statistics(self):
        # fitness 现在是归一化后的加权值，范围 [0, 1]，1.0 表示最差
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        valid_fitnesses = [f for f in fitnesses if f < 1.0]  # 排除失败的个体（fitness=1.0）
        
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
            'ntk_history': self.ntk_history,  # 保存NTK历史
            'search_time': self.search_time,  # 搜索时间
            'short_train_time': self.short_train_time,  # 短轮次训练时间
            'full_train_time': self.full_train_time,  # 完整训练时间
            'encoding_cache_keys': list(self.encoding_cache.keys()),  # 保存已评估的编码keys
            'duplicate_count': self.duplicate_count,  # 保存重复计数
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath} (unique encodings: {len(self.encoding_cache)}, duplicates: {self.duplicate_count})")

    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.population = deque(checkpoint['population'])
        self.history = checkpoint['history']
        # 加载NTK历史（兼容旧checkpoint）
        self.ntk_history = checkpoint.get('ntk_history', [])
        # 加载时间统计（兼容旧checkpoint）
        self.search_time = checkpoint.get('search_time', 0.0)
        self.short_train_time = checkpoint.get('short_train_time', 0.0)
        self.full_train_time = checkpoint.get('full_train_time', 0.0)
        # 加载重复计数（兼容旧checkpoint）
        self.duplicate_count = checkpoint.get('duplicate_count', 0)
        
        # 重建编码缓存：从history中重建
        self.encoding_cache = {}
        for ind in self.history:
            if ind.encoding and ind.fitness is not None:
                self._register_individual(ind)
        
        logger.info(f"Checkpoint loaded from {filepath} (unique encodings: {len(self.encoding_cache)}, duplicates: {self.duplicate_count})")
    
    def _save_ntk_history(self, filepath: str = None):
        """
        保存NTK历史记录到JSON文件
        """
        if not self.ntk_history:
            return
            
        if filepath is None:
            if not os.path.exists(config.LOG_DIR): 
                os.makedirs(config.LOG_DIR)
            filepath = os.path.join(config.LOG_DIR, 'ntk_history.json')
        
        # 转换为可序列化格式
        data = []
        for record in self.ntk_history:
            # 兼容旧格式 (4元素) 和新格式 (6元素)
            if len(record) == 4:
                step, ind_id, ntk_value, encoding = record
                fitness, param_count = None, None
            else:
                step, ind_id, ntk_value, fitness, param_count, encoding = record
            data.append({
                'step': step,
                'individual_id': ind_id,
                'ntk': ntk_value if ntk_value is not None else None,
                'fitness': fitness if fitness is not None else None,
                'param_count': param_count if param_count is not None else None,
                'encoding': encoding
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"NTK history saved to {filepath}")
    
    def plot_ntk_curve(self, output_path: str = None):
        """
        绘制搜索过程中NTK值的变化曲线
        包括：
        1. 所有个体的NTK散点图
        2. 滑动窗口平均NTK曲线
        3. 当前种群最佳NTK曲线
        """
        if not self.ntk_history:
            logger.warning("No NTK history to plot!")
            return
        
        if output_path is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            output_path = os.path.join(config.LOG_DIR, 'ntk_curve.png')
        
        # 提取数据（兼容新旧格式）
        steps = []
        ntk_values = []
        for record in self.ntk_history:
            if len(record) == 4:  # 旧格式
                step, ind_id, ntk, encoding = record
            else:  # 新格式
                step, ind_id, ntk, fitness, param_count, encoding = record
            if ntk is not None and ntk < 100000:  # 排除无效值
                steps.append(step)
                ntk_values.append(ntk)
        
        if not steps:
            logger.warning("No valid NTK values to plot!")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 所有个体的NTK散点图
        ax1 = axes[0, 0]
        ax1.scatter(steps, ntk_values, alpha=0.3, s=10, c='blue')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('NTK Condition Number')
        ax1.set_title('All Individuals NTK Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. 滑动窗口平均NTK曲线
        ax2 = axes[0, 1]
        window_size = max(10, len(ntk_values) // 50)  # 动态窗口大小
        if len(ntk_values) >= window_size:
            moving_avg = []
            for i in range(len(ntk_values) - window_size + 1):
                avg = sum(ntk_values[i:i+window_size]) / window_size
                moving_avg.append(avg)
            moving_avg_steps = steps[window_size-1:]
            ax2.plot(moving_avg_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            ax2.scatter(steps, ntk_values, alpha=0.2, s=5, c='blue', label='Individual NTK')
            ax2.legend()
        else:
            ax2.scatter(steps, ntk_values, alpha=0.5, s=10, c='blue')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('NTK Condition Number')
        ax2.set_title('NTK with Moving Average')
        ax2.grid(True, alpha=0.3)
        
        # 3. 按step分组的最佳NTK曲线
        ax3 = axes[1, 0]
        step_best = {}
        for record in self.ntk_history:
            if len(record) == 4:  # 旧格式
                step, ind_id, ntk, encoding = record
            else:  # 新格式
                step, ind_id, ntk, fitness, param_count, encoding = record
            if ntk is not None and ntk < 100000:
                if step not in step_best or ntk < step_best[step]:
                    step_best[step] = ntk
        
        sorted_steps = sorted(step_best.keys())
        best_ntks = [step_best[s] for s in sorted_steps]
        
        # 累积最佳
        cumulative_best = []
        current_best = float('inf')
        for ntk in best_ntks:
            current_best = min(current_best, ntk)
            cumulative_best.append(current_best)
        
        ax3.plot(sorted_steps, best_ntks, 'g-', alpha=0.5, label='Best per Step')
        ax3.plot(sorted_steps, cumulative_best, 'r-', linewidth=2, label='Cumulative Best')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('NTK Condition Number')
        ax3.set_title('Best NTK Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. NTK分布直方图
        ax4 = axes[1, 1]
        ax4.hist(ntk_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(min(ntk_values), color='r', linestyle='--', linewidth=2, label=f'Best: {min(ntk_values):.2f}')
        ax4.axvline(sum(ntk_values)/len(ntk_values), color='g', linestyle='--', linewidth=2, label=f'Mean: {sum(ntk_values)/len(ntk_values):.2f}')
        ax4.set_xlabel('NTK Condition Number')
        ax4.set_ylabel('Count')
        ax4.set_title('NTK Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"NTK curve saved to {output_path}")
        
        # 打印统计信息
        logger.info(f"NTK Statistics: Total={len(ntk_values)}, Best={min(ntk_values):.4f}, "
                   f"Mean={sum(ntk_values)/len(ntk_values):.4f}, Worst={max(ntk_values):.4f}")

    def plot_fitness_curve(self, output_path: str = None):
        """
        绘制搜索过程中Fitness值的变化曲线
        包括：
        1. 所有个体的Fitness散点图
        2. 滑动窗口平均Fitness曲线
        3. 累积最佳Fitness曲线
        4. Fitness分布直方图
        """
        if not self.ntk_history:
            logger.warning("No history to plot fitness!")
            return
        
        if output_path is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            output_path = os.path.join(config.LOG_DIR, 'fitness_curve.png')
        
        # 提取数据（仅支持新格式，旧格式无 fitness）
        steps = []
        fitness_values = []
        param_counts = []
        for record in self.ntk_history:
            if len(record) == 6:  # 新格式
                step, ind_id, ntk, fitness, param_count, encoding = record
                if fitness is not None and fitness < 1.0:  # 排除无效值
                    steps.append(step)
                    fitness_values.append(fitness)
                    if param_count is not None:
                        param_counts.append(param_count)
        
        if not steps:
            logger.warning("No valid Fitness values to plot! (需要新格式的历史记录)")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fitness Evolution During Search', fontsize=14, fontweight='bold')
        
        # 1. 所有个体的Fitness散点图
        ax1 = axes[0, 0]
        ax1.scatter(steps, fitness_values, alpha=0.3, s=10, c='purple')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Fitness (lower is better)')
        ax1.set_title('All Individuals Fitness Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. 滑动窗口平均Fitness曲线
        ax2 = axes[0, 1]
        window_size = max(10, len(fitness_values) // 50)  # 动态窗口大小
        if len(fitness_values) >= window_size:
            moving_avg = []
            for i in range(len(fitness_values) - window_size + 1):
                avg = sum(fitness_values[i:i+window_size]) / window_size
                moving_avg.append(avg)
            moving_avg_steps = steps[window_size-1:]
            ax2.plot(moving_avg_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            ax2.scatter(steps, fitness_values, alpha=0.2, s=5, c='purple', label='Individual Fitness')
            ax2.legend()
        else:
            ax2.scatter(steps, fitness_values, alpha=0.5, s=10, c='purple')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Fitness (lower is better)')
        ax2.set_title('Fitness with Moving Average')
        ax2.grid(True, alpha=0.3)
        
        # 3. 按step分组的最佳Fitness曲线
        ax3 = axes[1, 0]
        step_best = {}
        for record in self.ntk_history:
            if len(record) == 6:  # 新格式
                step, ind_id, ntk, fitness, param_count, encoding = record
                if fitness is not None and fitness < 1.0:
                    if step not in step_best or fitness < step_best[step]:
                        step_best[step] = fitness
        
        sorted_steps = sorted(step_best.keys())
        best_fitness_per_step = [step_best[s] for s in sorted_steps]
        
        # 累积最佳
        cumulative_best = []
        current_best = float('inf')
        for f in best_fitness_per_step:
            current_best = min(current_best, f)
            cumulative_best.append(current_best)
        
        ax3.plot(sorted_steps, best_fitness_per_step, 'g-', alpha=0.5, label='Best per Step')
        ax3.plot(sorted_steps, cumulative_best, 'r-', linewidth=2, label='Cumulative Best')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Fitness (lower is better)')
        ax3.set_title('Best Fitness Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Fitness分布直方图
        ax4 = axes[1, 1]
        ax4.hist(fitness_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
        best_fitness = min(fitness_values)
        mean_fitness = sum(fitness_values) / len(fitness_values)
        ax4.axvline(best_fitness, color='r', linestyle='--', linewidth=2, label=f'Best: {best_fitness:.4f}')
        ax4.axvline(mean_fitness, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean_fitness:.4f}')
        ax4.set_xlabel('Fitness (lower is better)')
        ax4.set_ylabel('Count')
        ax4.set_title('Fitness Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Fitness curve saved to {output_path}")
        
        # 打印统计信息
        logger.info(f"Fitness Statistics: Total={len(fitness_values)}, Best={best_fitness:.6f}, "
                   f"Mean={mean_fitness:.6f}, Worst={max(fitness_values):.6f}")

    def plot_param_fitness_correlation(self, output_path: str = None):
        """
        绘制参数量与Fitness的相关性图
        包括：
        1. 参数量 vs Fitness 散点图
        2. 参数量 vs NTK 散点图
        3. 参数量分布直方图
        4. 三维关系图 (参数量, NTK, Fitness)
        """
        if not self.ntk_history:
            logger.warning("No history to plot!")
            return
        
        if output_path is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            output_path = os.path.join(config.LOG_DIR, 'param_fitness_correlation.png')
        
        # 提取数据（仅支持新格式）
        param_counts = []
        fitness_values = []
        ntk_values = []
        for record in self.ntk_history:
            if len(record) == 6:
                step, ind_id, ntk, fitness, param_count, encoding = record
                if fitness is not None and param_count is not None and ntk is not None:
                    if fitness < 1.0 and ntk < 100000:
                        param_counts.append(param_count / 1e6)  # 转为百万
                        fitness_values.append(fitness)
                        ntk_values.append(ntk)
        
        if not param_counts:
            logger.warning("No valid data for correlation plot!")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Parameter Count vs Fitness/NTK Correlation', fontsize=14, fontweight='bold')
        
        # 1. 参数量 vs Fitness 散点图
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(param_counts, fitness_values, alpha=0.5, s=15, c=ntk_values, cmap='viridis')
        ax1.set_xlabel('Parameter Count (M)')
        ax1.set_ylabel('Fitness (lower is better)')
        ax1.set_title('Params vs Fitness (color=NTK)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='NTK')
        
        # 2. 参数量 vs NTK 散点图
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(param_counts, ntk_values, alpha=0.5, s=15, c=fitness_values, cmap='plasma')
        ax2.set_xlabel('Parameter Count (M)')
        ax2.set_ylabel('NTK Condition Number')
        ax2.set_title('Params vs NTK (color=Fitness)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Fitness')
        
        # 3. 参数量分布直方图
        ax3 = axes[1, 0]
        ax3.hist(param_counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
        mean_params = sum(param_counts) / len(param_counts)
        ax3.axvline(mean_params, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_params:.2f}M')
        ax3.set_xlabel('Parameter Count (M)')
        ax3.set_ylabel('Count')
        ax3.set_title('Parameter Count Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. NTK vs Fitness 散点图（颜色=参数量）
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(ntk_values, fitness_values, alpha=0.5, s=15, c=param_counts, cmap='coolwarm')
        ax4.set_xlabel('NTK Condition Number')
        ax4.set_ylabel('Fitness (lower is better)')
        ax4.set_title('NTK vs Fitness (color=Params)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Params (M)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Param-Fitness correlation plot saved to {output_path}")

    def _format_time(self, seconds: float) -> str:
        """
        将秒数格式化为可读的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}min ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f}h ({minutes:.0f}min)"

    def _save_time_stats(self, filepath: str = None):
        """
        保存时间统计到JSON文件并打印总结
        """
        total_time = self.search_time + self.short_train_time + self.full_train_time
        
        self.time_stats = {
            'search_phase': {
                'time_seconds': self.search_time,
                'time_formatted': self._format_time(self.search_time),
                'description': f'搜索阶段 (NTK评估 {self.max_gen} 个个体)'
            },
            'short_training_phase': {
                'time_seconds': self.short_train_time,
                'time_formatted': self._format_time(self.short_train_time),
                'description': f'短轮次训练阶段 (Top {config.HISTORY_TOP_N1} 个模型, {config.SHORT_TRAIN_EPOCHS} epochs)'
            },
            'full_training_phase': {
                'time_seconds': self.full_train_time,
                'time_formatted': self._format_time(self.full_train_time),
                'description': f'完整训练阶段 (Top {config.HISTORY_TOP_N2} 个模型, {config.FULL_TRAIN_EPOCHS} epochs)'
            },
            'total': {
                'time_seconds': total_time,
                'time_formatted': self._format_time(total_time),
                'description': '总耗时'
            }
        }
        
        # 保存到JSON文件
        if filepath is None:
            if not os.path.exists(config.LOG_DIR):
                os.makedirs(config.LOG_DIR)
            filepath = os.path.join(config.LOG_DIR, 'time_stats.json')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.time_stats, f, indent=2, ensure_ascii=False)
        
        # 打印时间统计总结
        logger.info("=" * 60)
        logger.info("时间统计总结")
        logger.info("=" * 60)
        logger.info(f"搜索阶段:       {self._format_time(self.search_time)}")
        logger.info(f"短轮次训练:     {self._format_time(self.short_train_time)}")
        logger.info(f"完整训练:       {self._format_time(self.full_train_time)}")
        logger.info("-" * 60)
        logger.info(f"总耗时:         {self._format_time(total_time)}")
        logger.info("=" * 60)
        logger.info(f"Time stats saved to {filepath}")
