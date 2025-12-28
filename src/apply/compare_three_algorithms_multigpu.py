# -*- coding: utf-8 -*-
"""
多GPU版本：比较三种搜索算法的性能差异
支持使用多张GPU（如2张4090）进行并行训练

使用方式：
    python compare_three_algorithms_multigpu.py --quick_test
    python compare_three_algorithms_multigpu.py --gpus 0,1

特点：
1. 使用DataParallel实现多GPU并行训练
2. 自动检测可用GPU数量
3. 支持指定使用的GPU设备
"""

import os
import sys
import json
import time
import random
import argparse
import warnings

# 抑制 NumPy 弃用警告（torchvision 与 NumPy 2.4+ 兼容性问题）
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*dtype.*align.*')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import deque
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
from scipy.spatial import ConvexHull
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.encoding import Encoder, Individual
from core.search_space import population_initializer
from search.mutation import mutation_operator, selection_operator, crossover_operator
from engine.evaluator import fitness_evaluator, clear_gpu_memory
from models.network import NetworkBuilder
from utils.logger import logger
from data.dataset import DatasetLoader

# 设置英文字体避免中文问题
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = True


class MultiGPUTrainer:
    """
    多GPU网络训练器
    使用DataParallel实现多GPU并行训练
    """
    
    def __init__(self, gpu_ids: List[int] = None):
        """
        Args:
            gpu_ids: 要使用的GPU ID列表，如[0, 1]表示使用GPU 0和GPU 1
                    如果为None，则自动使用所有可用GPU
        """
        # 更健壮地检测 CUDA 可用性
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            cuda_available = False
            
        if cuda_available:
            if gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = gpu_ids
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
            self.num_gpus = len(self.gpu_ids)
        else:
            self.gpu_ids = []
            self.device = torch.device('cpu')
            self.num_gpus = 0
            logger.warning("CUDA not available, falling back to CPU mode")
            
        logger.info(f"MultiGPUTrainer initialized with {self.num_gpus} GPU(s): {self.gpu_ids}")
        
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """使用DataParallel包装模型以支持多GPU"""
        model = model.to(self.device)
        
        if self.num_gpus > 1:
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
            logger.info(f"Model wrapped with DataParallel on GPUs: {self.gpu_ids}")
            
        return model
    
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """解包DataParallel模型"""
        if isinstance(model, nn.DataParallel):
            return model.module
        return model
    
    def train_one_epoch(self, model: nn.Module, trainloader: DataLoader,
                       criterion: nn.Module, optimizer: optim.Optimizer,
                       epoch: int, total_epochs: int) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(trainloader):
                progress = (batch_idx + 1) / len(trainloader) * 100
                acc = 100. * correct / total
                print(f'\r  [Epoch {epoch}/{total_epochs}] Batch: {batch_idx + 1}/{len(trainloader)} '
                      f'({progress:.1f}%) | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {acc:.2f}%', 
                      end='', flush=True)
        print()
        
        avg_loss = running_loss / len(trainloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self, model: nn.Module, testloader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """评估模型"""
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = test_loss / len(testloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train_network(self, model: nn.Module, trainloader: DataLoader,
                     testloader: DataLoader, epochs: int = None,
                     lr: float = None, momentum: float = None,
                     weight_decay: float = None) -> Tuple[float, List[dict]]:
        """
        训练网络（支持多GPU）
        
        Returns:
            best_acc: 最佳测试精度
            history: 训练历史
        """
        if epochs is None: epochs = config.FULL_TRAIN_EPOCHS
        if lr is None: lr = config.LEARNING_RATE
        if momentum is None: momentum = config.MOMENTUM
        if weight_decay is None: weight_decay = config.WEIGHT_DECAY
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 包装模型以支持多GPU
        model = self._wrap_model(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, 
                             momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = []
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self._unwrap_model(model).state_dict())
        
        try:
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = self.train_one_epoch(
                    model, trainloader, criterion, optimizer, epoch, epochs
                )
                test_loss, test_acc = self.evaluate(model, testloader, criterion)
                scheduler.step()
                
                history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model_wts = copy.deepcopy(self._unwrap_model(model).state_dict())
                
                print(f'  [Epoch {epoch}/{epochs}] Train Acc: {train_acc:.2f}% | '
                      f'Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%')
                      
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("OOM during training. Clearing cache.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 解包并恢复最佳权重
        model = self._unwrap_model(model)
        model.load_state_dict(best_model_wts)
        
        return best_acc, history


class MultiGPUFinalEvaluator:
    """
    多GPU版本的最终评估器
    """
    
    def __init__(self, dataset: str = 'cifar10', gpu_ids: List[int] = None, 
                 batch_size: int = None):
        """
        Args:
            dataset: 数据集名称
            gpu_ids: GPU ID列表
            batch_size: 批大小（多GPU时可以增大）
        """
        self.dataset = dataset
        self.gpu_ids = gpu_ids
        self.trainer = MultiGPUTrainer(gpu_ids)
        
        # 多GPU时可以使用更大的batch size
        if batch_size is None:
            num_gpus = len(self.trainer.gpu_ids) if self.trainer.gpu_ids else 1
            batch_size = config.BATCH_SIZE * max(1, num_gpus)
        
        self.batch_size = batch_size
        logger.info(f"Using batch size: {self.batch_size}")
        
        if dataset == 'cifar10':
            self.trainloader, self.testloader = DatasetLoader.get_cifar10(
                batch_size=self.batch_size
            )
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.trainloader, self.testloader = DatasetLoader.get_cifar100(
                batch_size=self.batch_size
            )
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def evaluate_individual(self, individual: Individual, epochs: int = None) -> Tuple[float, dict]:
        """评估个体"""
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS

        logger.info(f"Training individual {individual.id} for {epochs} epochs on {len(self.trainer.gpu_ids)} GPU(s)...")
        
        network = NetworkBuilder.build_from_individual(
            individual, input_channels=3, num_classes=self.num_classes
        )
        param_count = network.get_param_count()
        
        start_time = time.time()
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader, epochs
        )
        train_time = time.time() - start_time

        result = {
            'individual_id': individual.id,
            'encoding': individual.encoding,
            'param_count': param_count,
            'best_acc': best_acc,
            'train_time': train_time,
            'epochs': epochs,
            'history': history
        }
        
        logger.info(f"Individual {individual.id}: Best Acc = {best_acc:.2f}%, "
                   f"Params = {param_count:,}, Time = {train_time:.1f}s")
        
        # 清理内存
        del network
        clear_gpu_memory()
        
        return best_acc, result


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
    三阶段EA算法（多GPU版本）
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
    
    def run(self, evaluator: MultiGPUFinalEvaluator = None) -> List[ModelInfo]:
        """运行三阶段EA搜索"""
        logger.info("=" * 60)
        logger.info("Three-Stage EA Algorithm Started (Multi-GPU)")
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
    传统EA算法（经典Aging Evolution，多GPU版本）
    """
    
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
        """锦标赛选择"""
        current_pop_list = list(self.population)
        tournament = random.sample(current_pop_list, min(config.TOURNAMENT_SIZE, len(current_pop_list)))
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
    
    def _evaluate_fitness(self, ind: Individual, evaluator: MultiGPUFinalEvaluator) -> float:
        """评估适应度"""
        try:
            acc, _ = evaluator.evaluate_individual(ind, epochs=self.search_epochs)
            return acc
        except Exception as e:
            logger.warning(f"  Fitness evaluation failed: {e}")
            return 0.0
    
    def run(self, evaluator: MultiGPUFinalEvaluator = None) -> List[ModelInfo]:
        """运行传统EA搜索"""
        logger.info("=" * 60)
        logger.info("Traditional EA (Classic Aging Evolution) Started (Multi-GPU)")
        logger.info(f"Config: max_eval={self.max_evaluations}, pop={self.population_size}")
        logger.info(f"        search_epochs={self.search_epochs}, full_epochs={self.full_epochs}")
        logger.info(f"        top_n={self.top_n}")
        logger.info("=" * 60)
        
        eval_count = 0
        best_fitness = 0.0
        
        # ==================== Stage 1: Aging Evolution Search ====================
        logger.info(f"\n[Stage 1] Aging Evolution Search (fitness={self.search_epochs} epoch accuracy)...")
        
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
                logger.info(f"    Evolution progress: {eval_count}/{self.max_evaluations}, "
                           f"Current: {fitness:.2f}%, Best: {best_fitness:.2f}%")
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
            logger.info(f"  Full training [{i+1}/{len(top_candidates)}] ID: {ind.id}, "
                       f"Search Acc: {ind.fitness:.2f}%")
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
    随机搜索算法（多GPU版本）
    """
    
    def __init__(self, 
                 num_samples: int = 10,
                 full_epochs: int = 150):
        self.num_samples = num_samples
        self.full_epochs = full_epochs
        self.final_models: List[ModelInfo] = []
        
    def run(self, evaluator: MultiGPUFinalEvaluator = None) -> List[ModelInfo]:
        """运行随机搜索"""
        logger.info("=" * 60)
        logger.info("Random Search Started (Multi-GPU)")
        logger.info(f"Config: num_samples={self.num_samples}, full_epochs={self.full_epochs}")
        logger.info("=" * 60)
        
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
    """绘制三种算法的帕累托前沿对比图"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def extract_data(models: List[ModelInfo]):
        params = [m.param_count for m in models if m.param_count > 0 and m.accuracy > 0]
        accs = [m.accuracy for m in models if m.param_count > 0 and m.accuracy > 0]
        return np.array(params), np.array(accs)
    
    ts_params, ts_accs = extract_data(three_stage_models)
    te_params, te_accs = extract_data(traditional_models)
    rs_params, rs_accs = extract_data(random_models)
    
    colors = {
        'three_stage': '#FF6B6B',
        'traditional': '#98D8AA',
        'random': '#6B8EFF'
    }
    
    alpha_hull = 0.25
    alpha_scatter = 0.8
    
    def plot_with_hull(params, accs, color, label, marker='o'):
        if len(params) < 3:
            ax.scatter(params, accs, c=color, label=label, s=80, 
                      alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
            return
        
        ax.scatter(params, accs, c=color, label=label, s=80, 
                  alpha=alpha_scatter, edgecolors='white', linewidths=1, marker=marker)
        
        try:
            points = np.column_stack([params, accs])
            hull = ConvexHull(points)
            
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=color, alpha=alpha_hull)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 
                   color=color, linewidth=2, alpha=0.7)
        except Exception as e:
            logger.warning(f"Cannot draw convex hull: {e}")
    
    if len(ts_params) > 0:
        plot_with_hull(ts_params, ts_accs, colors['three_stage'], 'Three-Stage EA', 'o')
    if len(te_params) > 0:
        plot_with_hull(te_params, te_accs, colors['traditional'], 'Traditional EA', 's')
    if len(rs_params) > 0:
        plot_with_hull(rs_params, rs_accs, colors['random'], 'Random Search', '^')
    
    ax.set_xlabel('Parameters (M)', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Comparison of Three Search Algorithms (Multi-GPU)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
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
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'algorithm_comparison_multigpu_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chart saved to: {output_path}")
        
        pdf_path = os.path.join(output_dir, f'algorithm_comparison_multigpu_{timestamp}.pdf')
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
                'avg_accuracy': float(np.mean([m.accuracy for m in three_stage_models])) if three_stage_models else 0,
                'max_accuracy': float(max([m.accuracy for m in three_stage_models])) if three_stage_models else 0,
                'avg_params': float(np.mean([m.param_count for m in three_stage_models])) if three_stage_models else 0,
            },
            'traditional_ea': {
                'count': len(traditional_models),
                'avg_accuracy': float(np.mean([m.accuracy for m in traditional_models])) if traditional_models else 0,
                'max_accuracy': float(max([m.accuracy for m in traditional_models])) if traditional_models else 0,
                'avg_params': float(np.mean([m.param_count for m in traditional_models])) if traditional_models else 0,
            },
            'random_search': {
                'count': len(random_models),
                'avg_accuracy': float(np.mean([m.accuracy for m in random_models])) if random_models else 0,
                'max_accuracy': float(max([m.accuracy for m in random_models])) if random_models else 0,
                'avg_params': float(np.mean([m.param_count for m in random_models])) if random_models else 0,
            }
        }
    }
    
    output_path = os.path.join(output_dir, f'experiment_results_multigpu_{timestamp}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    return output_path


def run_experiment(args):
    """运行完整的对比实验"""
    
    # 解析GPU ID列表
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = None  # 使用所有可用GPU
    
    logger.info("=" * 70)
    logger.info("         Three Algorithm Comparison Experiment (Multi-GPU)")
    logger.info("=" * 70)
    
    # 显示GPU信息
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception as e:
        logger.warning(f"CUDA initialization check failed: {e}")
        
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        if gpu_ids:
            logger.info(f"Using GPUs: {gpu_ids}")
        else:
            logger.info(f"Using all {num_gpus} GPUs")
    else:
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"\nExperiment Config:")
    logger.info(f"  - Three-Stage EA: NTK evals={args.ts_ntk_evals}, short={args.ts_short_epochs}ep, full={args.full_epochs}ep")
    logger.info(f"  - Traditional EA: search evals={args.te_evals}, search={args.te_search_epochs}ep, "
               f"Top{args.te_top_n} full={args.full_epochs}ep")
    logger.info(f"  - Random Search: samples={args.rs_samples}, full={args.full_epochs}ep")
    logger.info("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'experiment_results'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建多GPU评估器
    evaluator = MultiGPUFinalEvaluator(
        dataset=config.FINAL_DATASET,
        gpu_ids=gpu_ids
    )
    
    three_stage_models = []
    traditional_models = []
    random_models = []
    
    # 1. 运行三阶段EA
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
    
    # 2. 运行传统EA
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
    
    # 3. 运行随机搜索
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
    
    # 打印最终统计
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
            logger.info(f"  Mean accuracy: {np.mean(accs):.2f}%")
            logger.info(f"  Params range: {min(params):.2f}M - {max(params):.2f}M")
            logger.info(f"  Mean params: {np.mean(params):.2f}M")
    
    print_stats("Three-Stage EA", three_stage_models)
    print_stats("Traditional EA", traditional_models)
    print_stats("Random Search", random_models)
    
    logger.info("\n" + "=" * 70)
    logger.info("Experiment completed!")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Three Algorithm Comparison Experiment (Multi-GPU)')
    
    # GPU参数
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,1"). Default: use all available GPUs')
    
    # 三阶段EA参数
    parser.add_argument('--ts_ntk_evals', type=int, default=1000,
                        help='Three-Stage EA NTK evaluation count')
    parser.add_argument('--ts_pop_size', type=int, default=50,
                        help='Three-Stage EA population size')
    parser.add_argument('--ts_top_n1', type=int, default=20,
                        help='Three-Stage EA first round Top N1')
    parser.add_argument('--ts_top_n2', type=int, default=10,
                        help='Three-Stage EA second round Top N2')
    parser.add_argument('--ts_short_epochs', type=int, default=30,
                        help='Three-Stage EA short training epochs')
    
    # 传统EA参数
    parser.add_argument('--te_evals', type=int, default=20,
                        help='Traditional EA search evaluation count')
    parser.add_argument('--te_pop_size', type=int, default=5,
                        help='Traditional EA population size')
    parser.add_argument('--te_top_n', type=int, default=10,
                        help='Traditional EA Top N for full training')
    parser.add_argument('--te_search_epochs', type=int, default=30,
                        help='Traditional EA search training epochs')
    
    # 随机搜索参数
    parser.add_argument('--rs_samples', type=int, default=10,
                        help='Random search sample count')
    
    # 共享参数
    parser.add_argument('--full_epochs', type=int, default=100,
                        help='Full training epochs')
    
    # 控制参数
    parser.add_argument('--skip_three_stage', action='store_true',
                        help='Skip Three-Stage EA')
    parser.add_argument('--skip_traditional', action='store_true',
                        help='Skip Traditional EA')
    parser.add_argument('--skip_random', action='store_true',
                        help='Skip Random Search')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display the plot')
    
    # 快速测试模式
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (reduced evaluations and epochs)')
    
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
