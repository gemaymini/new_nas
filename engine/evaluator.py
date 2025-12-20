import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self,
                 input_size: Tuple[int, int, int] = None,
                 num_classes: int = None,
                 batch_size: int = None,
                 device: str = None,
                 recalbn: int = 0,
                 num_batch: int = 1):
        self.input_size = input_size or config.NTK_INPUT_SIZE
        self.num_classes = num_classes or config.NTK_NUM_CLASSES
        self.batch_size = batch_size or config.NTK_BATCH_SIZE
        self.device = device or config.DEVICE
        self.param_threshold = config.NTK_PARAM_THRESHOLD

        self.recalbn = recalbn        # 重置并重新统计 BN 的 batch 数
        self.num_batch = num_batch    # 使用多少个 batch 计算 NTK

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # 加载用于 NTK 计算的小数据集 loader
        self.trainloader = DatasetLoader.get_ntk_trainloader(
            batch_size=self.batch_size
        )

    def recal_bn(self, network: nn.Module, xloader, recal_batches: int, device):
        """重置 BN 统计并用若干 batch 重新统计（与 ntk.py 中一致）"""
        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.data.fill_(0)
                m.running_var.data.fill_(0)
                m.num_batches_tracked.data.zero_()
                m.momentum = None  # 使用 None 表示使用 batch mean/var

        network.train()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(xloader):
                if i >= recal_batches:
                    break
                inputs = inputs.to(device=device, non_blocking=True)
                _ = network(inputs)
        return network

    def compute_ntk_condition_number(self, network: nn.Module, xloader, num_batch: int = -1) -> float:
        """严格按照 ntk.py 中的 get_ntk_n 逻辑计算单个网络的 NTK 条件数"""
        device = self.device if self.device == 'cpu' else 0

        network.eval()  # 默认 eval 模式，与原始实现一致
        grads = []

        for i, (inputs, _) in enumerate(xloader):
            if num_batch > 0 and i >= num_batch:
                break
            inputs = inputs.to(device=device, non_blocking=True)

            network.zero_grad()
            logit = network(inputs)
            if isinstance(logit, tuple):
                logit = logit[1]  # 支持返回 (features, logits) 的网络

            for idx in range(inputs.size(0)):
                network.zero_grad()
                logit[idx:idx + 1].backward(torch.ones_like(logit[idx:idx + 1]), retain_graph=True)

                grad = []
                for name, p in network.named_parameters():
                    if 'weight' in name and p.grad is not None:
                        grad.append(p.grad.view(-1).detach().clone())

                if grad:
                    grads.append(torch.cat(grad))

                torch.cuda.empty_cache()

        if len(grads) == 0:
            return 100000.0  # 无效网络，惩罚大条件数

        grads_tensor = torch.stack(grads)  # (N, C)
        ntk = torch.einsum('nc,mc->nm', grads_tensor, grads_tensor)  # (N, N)

        try:
            eigenvalues = torch.linalg.eigh(ntk)[0]  # 升序
        except Exception:
            eigenvalues = torch.symeig(ntk, eigenvectors=False)

        cond = (eigenvalues[-1] / eigenvalues[0]).item()
        cond = np.nan_to_num(cond, nan=100000.0, posinf=100000.0, neginf=100000.0)

        # 清理显存
        del grads, grads_tensor, ntk, eigenvalues, logit
        clear_gpu_memory()

        return cond

    def compute_ntk_score(self, network: nn.Module, param_count: int = None) -> float:
        try:
            if param_count and param_count > self.param_threshold:
                logger.warning(f"Skipping NTK: params {param_count} > threshold {self.param_threshold}")
                return 100000.0  # 大模型惩罚大条件数（差）

            network = network.to(self.device)

            if self.recalbn > 0:
                network = self.recal_bn(network, self.trainloader, self.recalbn, self.device)

            cond = self.compute_ntk_condition_number(network, self.trainloader, num_batch=self.num_batch)

            return cond

        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return 100000.0  # 出错也给差分

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            network = NetworkBuilder.build_from_individual(
                individual,
                input_channels=self.input_size[0],
                num_classes=self.num_classes
            )
            param_count = network.get_param_count()
            individual.param_count = param_count

            score = self.compute_ntk_score(network, param_count)

            # 注意：条件数越小越好，所以 fitness 取负值（进化算法通常越大越好）
            fitness = -score
            individual.fitness = fitness

            del network
            clear_gpu_memory()

            logger.log_evaluation(individual.id, "NTK", fitness)
            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = -100000.0
            clear_gpu_memory()
            return -100000.0


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
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS

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

        save_dir = os.path.join(config.CHECKPOINT_DIR, 'final_models')
        os.makedirs(save_dir, exist_ok=True)
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
        logger.info(f"Model {individual.id} Architecture Encoding: {individual.encoding}")

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
        if top_k is None:
            top_k = config.HISTORY_TOP_N2
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS

        sorted_pop = sorted(population, key=lambda x: x.fitness or float('-inf'), reverse=True)
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

    def evaluate_individual(self, individual: Individual) -> float:
        """评估单个个体的 NTK Fitness"""
        return self.ntk_evaluator.evaluate_individual(individual)

    def evaluate_population_ntk(self, population: List[Individual], show_progress: bool = True):
        total = len(population)
        clear_gpu_memory()

        for idx, ind in enumerate(population):
            if show_progress:
                print(f"\r[NTK Eval] {idx+1}/{total}", end="", flush=True)
            self.ntk_evaluator.evaluate_individual(ind)
            # param_count is calculated inside ntk_evaluator.evaluate_individual

            if (idx + 1) % 5 == 0:
                clear_gpu_memory()

        if show_progress:
            print()
        clear_gpu_memory()


# 全局实例
fitness_evaluator = FitnessEvaluator()