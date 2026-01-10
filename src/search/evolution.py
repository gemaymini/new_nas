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
        
        # 模型去重：记录已见过的编码
        self.seen_encodings: set = set()
        self.duplicate_count = 0  # 重复模型计数
        
        # NTK历史记录，用于绘制NTK曲线
        # 格式: [(step, individual_id, ntk_value, encoding), ...]
        self.ntk_history: List[Tuple[int, int, float, list]] = []
        
        self.start_time = time.time()
        
        # 时间记录
        self.search_time = 0.0  # 搜索阶段时间（秒）
        self.short_train_time = 0.0  # 短轮次训练时间（秒）
        self.full_train_time = 0.0  # 完整训练时间（秒）
        self.time_stats: dict = {}  # 详细时间统计
        
        self._log_search_space_info()

    def _log_search_space_info(self):
        logger.info(config.get_search_space_summary())
        logger.info(f"Aging Evolution Config: Pop Size={self.population_size}, Total Gen={self.max_gen}")

    def _is_duplicate(self, encoding: List[int]) -> bool:
        """
        检查编码是否已存在（重复模型）
        """
        enc_tuple = tuple(encoding)
        if enc_tuple in self.seen_encodings:
            return True
        return False
    
    def _register_encoding(self, encoding: List[int]):
        """
        注册新的编码到已见集合
        """
        enc_tuple = tuple(encoding)
        self.seen_encodings.add(enc_tuple)

    def initialize_population(self):
        """
        Initialize the population with random individuals until queue is full.
        Duplicate individuals are skipped.
        """
        logger.info("Initializing population...")
        max_attempts_per_individual = 100  # 每个位置最多尝试次数，防止无限循环
        
        while len(self.population) < self.population_size:
            # 尝试生成不重复的个体
            attempts = 0
            ind = None
            while attempts < max_attempts_per_individual:
                ind = population_initializer.create_valid_individual()
                if ind is None:
                    # 无法生成有效个体，继续尝试
                    attempts += 1
                    continue
                if not self._is_duplicate(ind.encoding):
                    break
                self.duplicate_count += 1
                attempts += 1
                
            if ind is None or attempts >= max_attempts_per_individual:
                if ind is None:
                    logger.error("Failed to create valid individual, retrying...")
                    continue  # 重新开始外层循环
                logger.warning(f"Max attempts reached, using last generated individual")
            
            # 注册编码并评估
            self._register_encoding(ind.encoding)
            ind.id = len(self.population)
            fitness_evaluator.evaluate_individual(ind)
            self.population.append(ind)
            self.history.append(ind)
            
            # 记录NTK值
            step = 0  # 初始化阶段step=0
            self.ntk_history.append((step, ind.id, ind.fitness, ind.encoding.copy()))
            
            if len(self.population) % 10 == 0:
                logger.info(f"Initialized {len(self.population)}/{self.population_size} individuals (duplicates skipped: {self.duplicate_count})")

        logger.info(f"Population initialized. Size: {len(self.population)}, Duplicates skipped: {self.duplicate_count}")
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
        for _ in range(20):
            ind = mutation_operator.mutate(random.choice(parents))
            if Encoder.validate_encoding(ind.encoding): return ind
        return random.choice(parents).copy()

    def step(self) -> bool:
        """
        Perform one step of Aging Evolution:
        1. Select parents
        2. Generate child (skip if duplicate)
        3. Evaluate child
        4. Atomic update: Push child, Pop oldest
        
        Returns:
            bool: True if a valid (non-duplicate) child was generated, False otherwise
        """
        max_attempts = 50  # 最大尝试次数，防止无限循环
        attempts = 0
        child = None
        is_duplicate = True
        
        while attempts < max_attempts:
            # 1. Select Parents
            parent1, parent2 = self._select_parents()
            
            # 2. Generate Offspring
            child = self._generate_offspring(parent1, parent2)
            
            # 3. Check for duplicates
            if self._is_duplicate(child.encoding):
                self.duplicate_count += 1
                attempts += 1
                continue  # 重复模型，重新生成
            
            # 找到非重复模型，跳出循环
            is_duplicate = False
            break
        
        if is_duplicate:
            # 达到最大尝试次数且仍然是重复的，记录警告但不注册（不计入有效搜索）
            logger.warning(f"Max attempts ({max_attempts}) reached in step, all generated children were duplicates. Skipping this step.")
            return False  # 返回 False 表示本步无效
        
        # 注册编码
        self._register_encoding(child.encoding)
        child.id = len(self.history)  # Assign new ID based on total history
        
        # 4. Evaluate (Calculate NTK)
        fitness_evaluator.evaluate_individual(child)
        
        # 当前step（进化代数）
        current_step = len(self.history) - len(self.population) + 1
        
        # 记录NTK值
        self.ntk_history.append((current_step, child.id, child.fitness, child.encoding.copy()))
        
        # 5. Atomic Update
        with self.lock:
            # Remove oldest (head of deque)
            removed_ind = self.population.popleft()
            
            # Add new (tail of deque)
            self.population.append(child)
            
            # Add to history
            self.history.append(child)
            
        # Logging
        if len(self.history) % 10 == 0:
            logger.info(f"Step {len(self.history)-len(self.population)}/{self.max_gen}: Child Fitness={child.fitness:.4f} (duplicates skipped: {self.duplicate_count})")
            self._record_statistics()
            # 每10步保存一次NTK历史
            self._save_ntk_history()
        
        return True

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
        
        consecutive_failures = 0
        max_consecutive_failures = 100  # 连续失败上限，防止无限循环
        
        while len(self.history) - len(self.population) < self.max_gen:
            success = self.step()
            
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive duplicate failures ({max_consecutive_failures}). Stopping search.")
                    break
                continue
            else:
                consecutive_failures = 0  # 重置计数
            
            if (len(self.history) - len(self.population)) % 100 == 0:
                self.save_checkpoint()

        # 记录搜索阶段时间
        self.search_time = time.time() - search_start_time
        logger.info(f"Search completed. Search time: {self._format_time(self.search_time)}")
        logger.info(f"Search statistics: {len(self.history)} valid individuals evaluated, {self.duplicate_count} duplicates skipped, {len(self.seen_encodings)} unique architectures")
        self.save_checkpoint()
        
        # 搜索结束后绘制NTK曲线
        self._save_ntk_history()
        self.plot_ntk_curve()

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
            avg_fitness = float('inf')  # 无有效数据时设为无穷大
            best_fitness = float('inf')
            
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
            'seen_encodings': self.seen_encodings,  # 保存已见编码集合
            'duplicate_count': self.duplicate_count,  # 保存重复计数
            'search_time': self.search_time,  # 搜索时间
            'short_train_time': self.short_train_time,  # 短轮次训练时间
            'full_train_time': self.full_train_time,  # 完整训练时间
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.population = deque(checkpoint['population'])
        self.history = checkpoint['history']
        # 加载NTK历史（兼容旧checkpoint）
        self.ntk_history = checkpoint.get('ntk_history', [])
        # 加载去重状态（兼容旧checkpoint）
        self.seen_encodings = checkpoint.get('seen_encodings', set())
        self.duplicate_count = checkpoint.get('duplicate_count', 0)
        # 如果是旧的checkpoint没有seen_encodings，从history重建
        if not self.seen_encodings and self.history:
            for ind in self.history:
                self._register_encoding(ind.encoding)
            logger.info(f"Rebuilt seen_encodings from history: {len(self.seen_encodings)} unique encodings")
        # 加载时间统计（兼容旧checkpoint）
        self.search_time = checkpoint.get('search_time', 0.0)
        self.short_train_time = checkpoint.get('short_train_time', 0.0)
        self.full_train_time = checkpoint.get('full_train_time', 0.0)
        logger.info(f"Checkpoint loaded from {filepath}, duplicates skipped so far: {self.duplicate_count}")
    
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
        for step, ind_id, ntk_value, encoding in self.ntk_history:
            data.append({
                'step': step,
                'individual_id': ind_id,
                'ntk': ntk_value if ntk_value is not None else None,
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
        
        # 提取数据
        steps = []
        ntk_values = []
        for step, ind_id, ntk, encoding in self.ntk_history:
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
        for step, ind_id, ntk, encoding in self.ntk_history:
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
