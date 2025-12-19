# -*- coding: utf-8 -*-
"""
评估器模块
包含NTK评估、参数评估、快速评估和最终评估
"""
import torch
import torch.nn as nn
import numpy as np
import math
import gc
import time
import os
from typing import Tuple, List
from utils.config import config
from core.encoding import Individual, Encoder
from model.network import NetworkBuilder
from utils.logger import logger, failed_logger
from data.dataset import DatasetLoader
from engine.trainer import NetworkTrainer

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class NTKEvaluator:
    def __init__(self, input_size: Tuple[int, int, int] = None,
                 num_classes: int = None, batch_size: int = None,
                 device: str = None):
        self.input_size = input_size or config.NTK_INPUT_SIZE
        self.num_classes = num_classes or config.NTK_NUM_CLASSES
        self.batch_size = batch_size or config.NTK_BATCH_SIZE
        self.device = device or config.DEVICE
        self.param_threshold = config.NTK_PARAM_THRESHOLD
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            
    def compute_ntk_score(self, network: nn.Module, param_count: int = None) -> float:
        try:
            # 安全检查：如果参数量过大，直接跳过NTK计算，防止OOM
            if param_count and param_count > config.NTK_PARAM_THRESHOLD: # 使用配置中的阈值
                logger.warning(f"Skipping NTK computation: params {param_count} > threshold {config.NTK_PARAM_THRESHOLD}")
                return 0.0 # 或者返回一个极低的分数

            if param_count and param_count > config.FORCE_CPU_EVAL_THRESHOLD * 1000000:
                return self._compute_simplified_score(network)
            
            # Simple memory check placeholder
            
            network = network.to(self.device)
            network.eval()
            effective_batch_size = min(self.batch_size, 4)
            x = torch.randn(effective_batch_size, *self.input_size).to(self.device)
            
            with torch.no_grad():
                _ = network(x)
                
            return self._compute_ntk_efficient(network, x)
        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return self._compute_simplified_score(network)
        finally:
            clear_gpu_memory()

    def _compute_ntk_efficient(self, network: nn.Module, x: torch.Tensor) -> float:
        try:
            network.train()
            num_classes_sample = min(self.num_classes, 3)
            outputs = network(x)
            grad_norms = []
            
            for class_idx in range(num_classes_sample):
                network.zero_grad()
                loss = outputs[:, class_idx].sum()
                loss.backward(retain_graph=(class_idx < num_classes_sample - 1))
                
                total_grad_norm = 0.0
                for p in network.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                grad_norms.append(math.sqrt(total_grad_norm))
            
            if len(grad_norms) > 0:
                mean_grad = np.mean(grad_norms)
                std_grad = np.std(grad_norms) if len(grad_norms) > 1 else 0.0
                if mean_grad > 0:
                    stability = 1.0 / (1.0 + std_grad / mean_grad)
                    score = stability * (1.0 / (1.0 + np.log(mean_grad + 1)))
                else:
                    score = 0.0
            else:
                score = 0.0
            return max(0.0, min(1.0, score))
        except Exception:
            return self._compute_simplified_score(network)

    def _compute_simplified_score(self, network: nn.Module) -> float:
        try:
            network = network.to('cpu')
            network.eval()
            weight_stats = []
            for name, param in network.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    w = param.data.view(param.size(0), -1)
                    fro_norm = torch.norm(w, p='fro').item()
                    try:
                        u = torch.randn(w.size(1), 1)
                        for _ in range(3):
                            v = w @ u
                            v = v / (torch.norm(v) + 1e-8)
                            u = w.t() @ v
                            u = u / (torch.norm(u) + 1e-8)
                        spectral_norm = (w @ u).norm().item()
                    except:
                        spectral_norm = fro_norm
                    
                    if spectral_norm > 0:
                        ratio = fro_norm / (spectral_norm + 1e-8)
                        weight_stats.append(ratio)
            
            if weight_stats:
                mean_ratio = np.mean(weight_stats)
                score = 1.0 / (1.0 + np.log(mean_ratio + 1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception:
            return 0.5

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            network = NetworkBuilder.build_from_individual(
                individual, input_channels=self.input_size[0], num_classes=self.num_classes
            )
            param_count = network.get_param_count()
            individual.param_count = param_count
            score = self.compute_ntk_score(network, param_count)
            individual.fitness = score
            del network
            clear_gpu_memory()
            logger.log_evaluation(individual.id, "NTK", score)
            return score
        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = 0.0
            clear_gpu_memory()
            return 0.0

class ParameterEvaluator:
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        self.input_channels = input_channels
        self.num_classes = num_classes
    
    def count_parameters(self, individual: Individual) -> int:
        try:
            if individual.param_count is not None and individual.param_count > 0:
                return individual.param_count
            network = NetworkBuilder.build_from_individual(
                individual, input_channels=self.input_channels, num_classes=self.num_classes
            )
            param_count = network.get_param_count()
            individual.param_count = param_count
            del network
            return param_count
        except Exception:
            individual.param_count = float('inf')
            return float('inf')

class QuickEvaluator:
    def __init__(self):
        self.num_samples = config.PHASE2_QUICK_EVAL_SAMPLES
        self.num_epochs = config.PHASE2_QUICK_EVAL_EPOCHS
        self.batch_size = config.PHASE2_QUICK_EVAL_BATCH_SIZE
        self.device = config.DEVICE
        self.input_channels = config.NTK_INPUT_SIZE[0]
        self.num_classes = config.NTK_NUM_CLASSES
        self._train_data = None
        self._val_data = None
        
    def _load_small_dataset(self):
        if self._train_data is not None: return
        
        import torchvision
        import torchvision.transforms as transforms
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        full_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        
        train_indices = torch.randperm(len(full_train))[:self.num_samples].tolist()
        val_indices = torch.randperm(len(full_val))[:self.num_samples // 4].tolist()
        
        self._train_data = torch.utils.data.Subset(full_train, train_indices)
        self._val_data = torch.utils.data.Subset(full_val, val_indices)

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            self._load_small_dataset()
            network = NetworkBuilder.build_from_individual(
                individual, input_channels=self.input_channels, num_classes=self.num_classes
            )
            if individual.param_count is None:
                individual.param_count = network.get_param_count()
            
            if individual.param_count > config.FORCE_CPU_EVAL_THRESHOLD * 1000000:
                individual.quick_score = 0.0
                return 0.0
                
            network = network.to(self.device)
            train_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(self._val_data, batch_size=self.batch_size, shuffle=False)
            
            optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            network.train()
            for _ in range(self.num_epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = network(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
            network.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = network(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            individual.quick_score = accuracy
            del network, optimizer
            clear_gpu_memory()
            return accuracy
        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}")
            individual.quick_score = 0.0
            clear_gpu_memory()
            return 0.0

class FinalEvaluator:
    def __init__(self, dataset: str = 'cifar10', device: str = None):
        self.dataset = dataset
        self.device = device or config.DEVICE
        self.trainer = NetworkTrainer(self.device)
        
        if dataset == 'cifar10':
            self.trainloader, self.testloader = DatasetLoader.get_cifar10()
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.trainloader, self.testloader = DatasetLoader.get_cifar100()
            self.num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    def evaluate_individual(self, individual: Individual, epochs: int = None) -> Tuple[float, dict]:
        if epochs is None: epochs = config.FINAL_TRAIN_EPOCHS
        
        logger.info(f"Training individual {individual.id} for {epochs} epochs...")
        network = NetworkBuilder.build_from_individual(
            individual, input_channels=3, num_classes=self.num_classes
        )
        param_count = network.get_param_count()
        Encoder.print_architecture(individual.encoding)
        
        start_time = time.time()
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader, epochs
        )
        train_time = time.time() - start_time
        
        # Save trained model
        save_dir = os.path.join(config.CHECKPOINT_DIR, 'final_models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'model_{individual.id}_acc{best_acc:.2f}.pth')
        
        save_dict = {
            'state_dict': network.state_dict(),
            'encoding': individual.encoding,
            'accuracy': best_acc,
            'param_count': param_count,
            'history': history
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved model to {save_path}")
        
        result = {
            'individual_id': individual.id,
            'param_count': param_count,
            'best_accuracy': best_acc,
            'train_time': train_time,
            'history': history,
            'encoding': individual.encoding,
            'model_path': save_path
        }
        return best_acc, result

    def evaluate_top_individuals(self, population: List[Individual], 
                                top_k: int = None, epochs: int = None) -> Tuple[Individual, List[dict]]:
        if top_k is None: top_k = config.FINAL_TOP_K
        if epochs is None: epochs = config.FINAL_TRAIN_EPOCHS
        
        sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness else float('-inf'), reverse=True)
        top_individuals = sorted_pop[:top_k]
        
        results = []
        best_individual = None
        best_accuracy = 0.0
        
        for idx, individual in enumerate(top_individuals):
            print(f"\n[{idx + 1}/{top_k}] Evaluating Individual {individual.id}")
            acc, result = self.evaluate_individual(individual, epochs)
            results.append(result)
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_individual = individual
        
        return best_individual, results

class FitnessEvaluator:
    def __init__(self):
        self.ntk_evaluator = NTKEvaluator()
        self.param_evaluator = ParameterEvaluator()
        self.quick_evaluator = QuickEvaluator()
        
    def evaluate_population_ntk(self, population: List[Individual], show_progress: bool = True):
        total = len(population)
        clear_gpu_memory()
        
        for idx, ind in enumerate(population):
            if show_progress:
                print(f"\r[NTK Eval] {idx+1}/{total}", end="", flush=True)
            self.ntk_evaluator.evaluate_individual(ind)
            if ind.param_count is None:
                self.param_evaluator.count_parameters(ind)
            
            if (idx + 1) % 5 == 0:
                clear_gpu_memory()
                
        if show_progress: print()
        clear_gpu_memory()

    def evaluate_population_survival(self, population: List[Individual], 
                                   current_gen: int, quick_eval: bool = True):
        total = len(population)
        for idx, ind in enumerate[Individual](population):
            ind.survival_time = current_gen - ind.birth_generation
            if ind.param_count is None:
                self.param_evaluator.count_parameters(ind)
            
            if quick_eval and ind.quick_score is None:
                print(f"\r[Quick Eval] {idx+1}/{total}", end="", flush=True)
                self.quick_evaluator.evaluate_individual(ind)
            
            # Update fitness to be quick_score for statistics and best tracking in Phase 2
            if quick_eval and ind.quick_score is not None:
                ind.fitness = ind.quick_score
        if quick_eval: print()

fitness_evaluator = FitnessEvaluator()
