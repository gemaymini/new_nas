# -*- coding: utf-8 -*-
"""
比较三种搜索算法的性能差异：
1. 三阶段EA (NTK筛选 → 短期训练 → 完整训练)
2. 传统EA (直接进化 + 完整训练，不使用NTK代理)
3. 随机搜索 (随机采样 + 完整训练)

实验目标：
- 比较在相同搜索空间下，三种算法找到的模型在参数量和精度上的帕累托前沿
- 生成类似论文图3-13的可视化对比图

实验设计：
- 所有算法使用相同的搜索空间
- 每种算法选出多个候选模型进行完整训练
- 最终比较参数量 vs 验证精度的分布
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import deque
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from scipy.spatial import ConvexHull
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, FinalEvaluator, clear_gpu_memory
from models.network import NetworkBuilder
from utils.logger import logger

# 设置英文字体避免中文乱码
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = True


class ModelInfo:
    """存储模型信息的数据类"""
    def __init__(self, individual: Individual, param_count: float = 0, 
                 accuracy: float = 0, ntk_score: float = None):
        self.individual = individual
        self.param_count = param_count  # 参数量（单位：M，百万）
        self.accuracy = accuracy        # 验证精度（%）
        self.ntk_score = ntk_score      # NTK条件数
        
    def to_dict(self):
        return {
            'id': self.individual.id,
            'encoding': self.individual.encoding,
            'param_count': self.param_count,
            'accuracy': self.accuracy,
            'ntk_score': self.ntk_score
        }


def count_parameters(model) -> float:
    """计算模型参数量（单位：M，百万）"""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def get_model_param_count(individual: Individual) -> float:
    """根据个体编码获取模型参数量"""
    try:
        model = NetworkBuilder.build_from_individual(individual)
        return count_parameters(model)
    except Exception as e:
        logger.warning(f"Failed to build model for param count: {e}")
        return 0


class ThreeStageEA:
    """
    三阶段EA算法（用户的算法）
    阶段1: NTK筛选 - 使用老化进化搜索，以NTK条件数为适应度
    阶段2: 短期训练筛选 - 对Top N1进行短期训练
    阶段3: 完整训练 - 对Top N2进行完整训练
    """
    
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
        """锦标赛选择"""
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
        """生成后代"""
        child = None
        
        if random.random() < config.PROB_CROSSOVER:
            c1, c2 = crossover_operator.crossover(parent1, parent2)
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
    
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        """
        运行三阶段EA搜索
        
        Returns:
            final_models: 完整训练后的模型列表
        """
        logger.info("=" * 60)
        logger.info("Three-Stage EA Algorithm Started")
        logger.info(f"Config: max_eval={self.max_evaluations}, pop={self.population_size}")
        logger.info(f"        top_n1={self.top_n1}, top_n2={self.top_n2}")
        logger.info("=" * 60)
        
        # ==================== Stage 1: NTK Screening ====================
        logger.info("\n[Stage 1] NTK Evaluation & Aging Evolution Search...")
        
        eval_count = 0
        
        # 初始化种群
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
        
        # 老化进化循环
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
        
        # 去重并按NTK排序
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
    """
    传统EA算法（经典Aging Evolution）
    
    采用经典的老化进化算法：
    - 搜索阶段：以短期训练（30轮）后的验证精度作为适应度
    - 选择最优的若干模型进行完整训练
    
    与三阶段EA的区别：不使用NTK代理，直接用训练精度指导搜索
    """
    
    def __init__(self, 
                 max_evaluations: int = 100,
                 population_size: int = 20,
                 top_n: int = 10,
                 search_epochs: int = 30,
                 full_epochs: int = 150):
        self.max_evaluations = max_evaluations  # 搜索阶段评估次数
        self.population_size = population_size
        self.top_n = top_n  # 最终完整训练的模型数量
        self.search_epochs = search_epochs  # 搜索阶段训练轮数（适应度评估）
        self.full_epochs = full_epochs  # 完整训练轮数
        
        self.population = deque()
        self.history: List[Individual] = []
        self.final_models: List[ModelInfo] = []
        
    def _select_parents(self) -> Tuple[Individual, Individual]:
        """锦标赛选择（基于验证精度，精度越高越好）"""
        current_pop_list = list(self.population)
        tournament = random.sample(current_pop_list, min(config.TOURNAMENT_SIZE, len(current_pop_list)))
        # 按精度降序排列（精度越高越好）
        tournament.sort(key=lambda x: x.fitness if x.fitness is not None else 0, reverse=True)
        
        if len(tournament) < 2:
            return tournament[0], tournament[0]
        return tournament[0], tournament[1]
    
    def _generate_offspring(self, parent1: Individual, parent2: Individual) -> Individual:
        """生成后代"""
        child = None
        
        if random.random() < config.PROB_CROSSOVER:
            c1, c2 = crossover_operator.crossover(parent1, parent2)
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
    
    def _evaluate_fitness(self, ind: Individual, evaluator: FinalEvaluator) -> float:
        """
        评估个体适应度：训练search_epochs轮后的验证精度
        """
        try:
            acc, _ = evaluator.evaluate_individual(ind, epochs=self.search_epochs)
            return acc
        except Exception as e:
            logger.warning(f"  Fitness evaluation failed: {e}")
            return 0.0
    
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        """
        运行传统EA搜索（经典Aging Evolution）
        
        阶段1: 老化进化搜索，以短期训练精度为适应度
        阶段2: 对Top N模型进行完整训练
        
        Returns:
            final_models: 完整训练后的模型列表
        """
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
            
            # 评估适应度（短期训练）
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
            # 选择父代
            parent1, parent2 = self._select_parents()
            
            # 生成后代
            child = self._generate_offspring(parent1, parent2)
            child.id = eval_count
            
            # 评估适应度
            fitness = self._evaluate_fitness(child, evaluator)
            child.fitness = fitness
            
            # 老化进化核心：移除最老的个体，添加新个体
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
        
        # 去重并按适应度排序
        unique_history = {}
        for ind in self.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            elif ind.fitness is not None:
                if unique_history[enc_tuple].fitness is None or ind.fitness > unique_history[enc_tuple].fitness:
                    unique_history[enc_tuple] = ind
        
        candidates = list(unique_history.values())
        # 按精度降序排列
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
    """
    随机搜索算法
    随机采样架构并进行完整训练评估
    """
    
    def __init__(self, 
                 num_samples: int = 10,
                 full_epochs: int = 150):
        self.num_samples = num_samples
        self.full_epochs = full_epochs
        self.final_models: List[ModelInfo] = []
        
    def run(self, evaluator: FinalEvaluator = None) -> List[ModelInfo]:
        """
        运行随机搜索
        
        Returns:
            final_models: 完整训练后的模型列表
        """
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


def plot_pareto_comparison(three_stage_models: List[ModelInfo],
                           traditional_models: List[ModelInfo],
                           random_models: List[ModelInfo],
                           output_dir: str = None,
                           show_plot: bool = True):
    """
    绘制三种算法的帕累托前沿对比图
    
    类似论文图3-13：
    - X轴: 参数量 (M)
    - Y轴: 验证精度 (%)
    - 每种算法用不同颜色的点表示
    - 绘制凸包显示各算法的覆盖范围
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 准备数据
    def extract_data(models: List[ModelInfo]):
        params = [m.param_count for m in models if m.param_count > 0 and m.accuracy > 0]
        accs = [m.accuracy for m in models if m.param_count > 0 and m.accuracy > 0]
        return np.array(params), np.array(accs)
    
    ts_params, ts_accs = extract_data(three_stage_models)
    te_params, te_accs = extract_data(traditional_models)
    rs_params, rs_accs = extract_data(random_models)
    
    # 颜色配置
    colors = {
        'three_stage': '#FF6B6B',     # 红色 - 三阶段EA
        'traditional': '#98D8AA',      # 绿色 - 传统EA
        'random': '#6B8EFF'            # 蓝色 - 随机搜索
    }
    
    alpha_hull = 0.25
    alpha_scatter = 0.8
    
    def plot_with_hull(params, accs, color, label, marker='o'):
        """绘制散点图和凸包"""
        if len(params) < 3:
            # 点太少无法绘制凸包
            ax.scatter(params, accs, c=color, label=label, s=80, 
                      alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
            return
        
        # 绘制散点
        ax.scatter(params, accs, c=color, label=label, s=80, 
                  alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
        
        # 尝试绘制凸包
        try:
            points = np.column_stack([params, accs])
            hull = ConvexHull(points)
            
            # 绘制凸包边界和填充
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])  # 闭合
            
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=color, alpha=alpha_hull)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 
                   color=color, linewidth=2, alpha=0.7)
        except Exception as e:
            logger.warning(f"Cannot draw convex hull: {e}")
    
    # 绘制三种算法的结果
    if len(ts_params) > 0:
        plot_with_hull(ts_params, ts_accs, colors['three_stage'], 'Three-Stage EA', 'o')
    if len(te_params) > 0:
        plot_with_hull(te_params, te_accs, colors['traditional'], 'Traditional EA', 's')
    if len(rs_params) > 0:
        plot_with_hull(rs_params, rs_accs, colors['random'], 'Random Search', '^')
    
    # 设置图表样式
    ax.set_xlabel('Parameters (M)', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Comparison of Three Search Algorithms', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = []
    if len(ts_accs) > 0:
        stats_text.append(f"Three-Stage EA: Avg Acc={np.mean(ts_accs):.2f}%, Avg Params={np.mean(ts_params):.2f}M")
    if len(te_accs) > 0:
        stats_text.append(f"Traditional EA: Avg Acc={np.mean(te_accs):.2f}%, Avg Params={np.mean(te_params):.2f}M")
    if len(rs_accs) > 0:
        stats_text.append(f"Random Search: Avg Acc={np.mean(rs_accs):.2f}%, Avg Params={np.mean(rs_params):.2f}M")
    
    stats_str = '\n'.join(stats_text)
    ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'algorithm_comparison_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chart saved to: {output_path}")
        
        # Also save PDF format
        pdf_path = os.path.join(output_dir, f'algorithm_comparison_{timestamp}.pdf')
        plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
        logger.info(f"PDF saved to: {pdf_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def save_experiment_results(three_stage_models: List[ModelInfo],
                            traditional_models: List[ModelInfo],
                            random_models: List[ModelInfo],
                            output_dir: str,
                            config_dict: dict = None):
    """保存实验结果到JSON文件"""
    
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
    """运行完整的对比实验"""
    
    logger.info("=" * 70)
    logger.info("         Three Algorithm Comparison Experiment")
    logger.info("=" * 70)
    logger.info(f"Experiment Config:")
    logger.info(f"  - Three-Stage EA: NTK evals={args.ts_ntk_evals}, short={args.ts_short_epochs}ep, full={args.full_epochs}ep")
    logger.info(f"  - Traditional EA: search evals={args.te_evals}, search={args.te_search_epochs}ep, Top{args.te_top_n} full={args.full_epochs}ep")
    logger.info(f"  - Random Search: samples={args.rs_samples}, full={args.full_epochs}ep")
    logger.info("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'experiment_results'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建共享的评估器
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
    
    # 保存实验结果
    config_dict = vars(args)
    save_experiment_results(
        three_stage_models,
        traditional_models,
        random_models,
        output_dir,
        config_dict
    )
    
    # 绘制对比图
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
    parser = argparse.ArgumentParser(description='三种搜索算法对比实验')
    
    # 三阶段EA参数
    parser.add_argument('--ts_ntk_evals', type=int, default=1000,
                        help='三阶段EA的NTK评估次数')
    parser.add_argument('--ts_pop_size', type=int, default=50,
                        help='三阶段EA的种群大小')
    parser.add_argument('--ts_top_n1', type=int, default=20,
                        help='三阶段EA第一轮筛选Top N1')
    parser.add_argument('--ts_top_n2', type=int, default=10,
                        help='三阶段EA第二轮筛选Top N2')
    parser.add_argument('--ts_short_epochs', type=int, default=30,
                        help='三阶段EA短期训练轮数')
    
    # 传统EA参数
    parser.add_argument('--te_evals', type=int, default=15,
                        help='传统EA的搜索阶段评估次数')
    parser.add_argument('--te_pop_size', type=int, default=5,
                        help='传统EA的种群大小')
    parser.add_argument('--te_top_n', type=int, default=10,
                        help='传统EA完整训练的Top N模型数')
    parser.add_argument('--te_search_epochs', type=int, default=30,
                        help='传统EA搜索阶段训练轮数（适应度评估）')
    
    # 随机搜索参数
    parser.add_argument('--rs_samples', type=int, default=16,
                        help='随机搜索的采样数量')
    
    # 共享参数
    parser.add_argument('--full_epochs', type=int, default=100,
                        help='完整训练轮数')
    
    # 控制参数
    parser.add_argument('--skip_three_stage', action='store_true',
                        help='跳过三阶段EA')
    parser.add_argument('--skip_traditional', action='store_true',
                        help='跳过传统EA')
    parser.add_argument('--skip_random', action='store_true',
                        help='跳过随机搜索')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图表')
    
    # 快速测试模式
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式（减少评估次数和训练轮数）')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick_test:
        args.ts_ntk_evals = 50
        args.ts_pop_size = 10
        args.ts_top_n1 = 5
        args.ts_top_n2 = 3
        args.ts_short_epochs = 5
        args.te_evals = 20
        args.te_pop_size = 5
        args.te_top_n = 3
        args.te_search_epochs = 5
        args.rs_samples = 5
        args.full_epochs = 10
        logger.info("Using quick test mode")
    
    run_experiment(args)


if __name__ == '__main__':
    main()
