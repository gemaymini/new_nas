# -*- coding: utf-8 -*-
"""
DARTS-based Aging Evolution Algorithm Implementation
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
from core.encoding import Individual, Encoder
from core.search_space import population_initializer, search_space
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, FinalEvaluator
from utils.logger import logger, failed_logger


class AgingEvolutionNAS:
    def __init__(self):
        self.population_size = config.POPULATION_SIZE
        self.max_gen = config.MAX_GEN  # Total individuals to evaluate
        
        # 1. Population & History Management
        # Using deque for FIFO queue (fixed size handled by manual popleft)
        self.population = deque() 
        self.history: List[Individual] = []
        self.lock = threading.Lock()
        
        # NTK历史记录，用于绘制NTK曲线
        # 格式: [(step, individual_id, ntk_value, genotype), ...]
        self.ntk_history: List[Tuple[int, int, float, dict]] = []
        
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

    def initialize_population(self):
        """
        Initialize the population with random individuals until queue is full.
        """
        logger.info("Initializing population...")
        
        while len(self.population) < self.population_size:
            ind = population_initializer.create_valid_individual()
            if ind is None:
                logger.warning("Failed to create individual, using fallback")
                ind = search_space.sample_individual()
            
            # Evaluate immediately
            ind.id = len(self.population)
            fitness_evaluator.evaluate_individual(ind)
            self.population.append(ind)
            self.history.append(ind)
            
            # 记录NTK值 (保存genotype而非encoding)
            step = 0  # 初始化阶段step=0
            genotype = Encoder.get_genotype(ind)
            self.ntk_history.append((step, ind.id, ind.fitness, genotype))
            
            if len(self.population) % 10 == 0:
                logger.info(f"Initialized {len(self.population)}/{self.population_size} individuals")

        logger.info(f"Population initialized. Size: {len(self.population)}")
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
            child = crossover_operator.crossover(parent1, parent2)
        else:
            child = random.choice([parent1, parent2]).copy()

        # Mutation
        if random.random() < config.PROB_MUTATION:
            child = mutation_operator.mutate(child)
            
        # Validate and Repair
        if not child.validate():
            child = self._repair_individual(child, [parent1, parent2])
            
        return child

    def _repair_individual(self, ind: Individual, parents: List[Individual]) -> Individual:
        """Repair invalid individual by re-mutating from parents"""
        for _ in range(20):
            ind = mutation_operator.mutate(random.choice(parents))
            if ind.validate():
                return ind
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
        
        # 当前step（进化代数）
        current_step = len(self.history) - len(self.population) + 1
        
        # 记录NTK值
        genotype = Encoder.get_genotype(child)
        self.ntk_history.append((current_step, child.id, child.fitness, genotype))
        
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
        while len(self.history) - len(self.population) < self.max_gen:
            self.step()
            
            if (len(self.history) - len(self.population)) % 100 == 0:
                self.save_checkpoint()

        # 记录搜索阶段时间
        self.search_time = time.time() - search_start_time
        logger.info(f"Search completed. Search time: {self._format_time(self.search_time)}")
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
        # Deduplicate history first based on genotype
        unique_history = {}
        for ind in self.history:
            # Use genotype as key for deduplication
            genotype = Encoder.get_genotype(ind)
            key = (tuple(genotype['normal']), tuple(genotype['reduce']))
            
            if key not in unique_history:
                unique_history[key] = ind
            else:
                # fitness 越小越好，保留更小的
                if ind.fitness is not None and (unique_history[key].fitness is None or ind.fitness < unique_history[key].fitness):
                    unique_history[key] = ind
        
        candidates = list(unique_history.values())
        # fitness 越小越好，升序排列
        candidates.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'), reverse=False)
        
        top_n1 = candidates[:config.HISTORY_TOP_N1]
        logger.info(f"Selected Top {config.HISTORY_TOP_N1} candidates from {len(candidates)} unique history individuals based on NTK.")
        
        # 2. Short Training (Top N1 -> Val Acc)
        logger.info(f"Starting Short Training ({config.SHORT_TRAIN_EPOCHS} epochs) for Top {config.HISTORY_TOP_N1}...")
        short_train_start_time = time.time()
        
        evaluator = FinalEvaluator(dataset=config.FINAL_DATASET)
        
        short_results = []
        for i, ind in enumerate(top_n1):
            logger.info(f"Short Train [{i+1}/{len(top_n1)}] ID: {ind.id}")
            acc, _ = evaluator.evaluate_individual(ind, epochs=config.SHORT_TRAIN_EPOCHS)
            ind.accuracy = acc  # Store for sorting
            short_results.append(ind)
        
        # 记录短轮次训练时间
        self.short_train_time = time.time() - short_train_start_time
        logger.info(f"Short Training completed. Time: {self._format_time(self.short_train_time)}")
            
        # 3. Select Top N2 (by Val Acc)
        short_results.sort(key=lambda x: x.accuracy if x.accuracy else float('-inf'), reverse=True)
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
            avg_fitness = best_fitness = 0.0
            
        stats = {
            'generation': len(self.history) - len(self.population),
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'population_size': len(self.population)
        }
        
        # Use existing logger methods
        logger.log_generation(len(self.history) - len(self.population), best_fitness, avg_fitness, len(self.population))

    def save_checkpoint(self, filepath: str = None):
        if filepath is None:
            if not os.path.exists(config.CHECKPOINT_DIR):
                os.makedirs(config.CHECKPOINT_DIR)
            filepath = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_step{len(self.history)-len(self.population)}.pkl')
        
        # 将 Individual 转换为可序列化格式
        population_data = [ind.to_dict() for ind in self.population]
        history_data = [ind.to_dict() for ind in self.history]
        
        checkpoint = {
            'population': population_data,
            'history': history_data,
            'ntk_history': self.ntk_history,
            'search_time': self.search_time,
            'short_train_time': self.short_train_time,
            'full_train_time': self.full_train_time,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # 从序列化格式恢复 Individual
        population_data = checkpoint.get('population', [])
        history_data = checkpoint.get('history', [])
        
        # 兼容旧格式 (直接存储 Individual 对象)
        if population_data and isinstance(population_data[0], dict):
            self.population = deque([Individual.from_dict(d) for d in population_data])
            self.history = [Individual.from_dict(d) for d in history_data]
        else:
            # 旧格式兼容
            self.population = deque(population_data)
            self.history = history_data
        
        # 加载NTK历史（兼容旧checkpoint）
        self.ntk_history = checkpoint.get('ntk_history', [])
        # 加载时间统计（兼容旧checkpoint）
        self.search_time = checkpoint.get('search_time', 0.0)
        self.short_train_time = checkpoint.get('short_train_time', 0.0)
        self.full_train_time = checkpoint.get('full_train_time', 0.0)
        
        # 更新 Individual ID 计数器，避免 ID 冲突
        if self.history:
            max_id = max(ind.id for ind in self.history)
            Individual.update_id_counter(max_id)
            
        logger.info(f"Checkpoint loaded from {filepath}")
    
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
        for step, ind_id, ntk_value, genotype in self.ntk_history:
            item = {
                'step': step,
                'individual_id': ind_id,
                'ntk': ntk_value if ntk_value is not None else None,
            }
            # 处理 genotype（可能是 dict 或旧格式的 list）
            if isinstance(genotype, dict):
                item['genotype'] = {
                    'normal': genotype.get('normal', []),
                    'reduce': genotype.get('reduce', [])
                }
            else:
                item['encoding'] = genotype  # 旧格式兼容
            data.append(item)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"NTK history saved to {filepath}")
    
    def plot_ntk_curve(self, output_path: str = None):
        """
        绘制搜索过程中NTK值的变化曲线
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
        for step, ind_id, ntk, genotype in self.ntk_history:
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
        window_size = max(10, len(ntk_values) // 50)
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
        for step, ind_id, ntk, genotype in self.ntk_history:
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
