import torch
import torch.nn as nn
import numpy as np
import gc
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple, List
from configuration.config import config
from core.encoding import Individual, Encoder
from models.network import NetworkBuilder
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
                 num_batch: int = 1,
                 dataset: str = None):
        self.input_size = input_size or config.NTK_INPUT_SIZE
        self.num_classes = num_classes or config.NTK_NUM_CLASSES
        self.batch_size = batch_size or config.NTK_BATCH_SIZE
        self.device = device or config.DEVICE
        self.param_threshold = config.NTK_PARAM_THRESHOLD
        self.dataset = dataset or config.FINAL_DATASET

        self.recalbn = recalbn        # 重置并重新统计 BN 的 batch 数
        self.num_batch = num_batch    # 使用多少个 batch 计算 NTK

        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # 加载用于 NTK 计算的小数据集 loader
        self.trainloader = DatasetLoader.get_ntk_trainloader(
            batch_size=self.batch_size,
            dataset_name=self.dataset
        )
        
        # ==================== 熵权法相关 ====================
        # 历史数据用于动态计算熵权重
        self.history_ntk: List[float] = []      # 历史 NTK 值
        self.history_param: List[float] = []    # 历史参数量
        self.entropy_weights = None              # 熵权法计算的权重 (w_ntk, w_param)
        self.dynamic_normalize_range = None      # 动态归一化范围
        
        # 熵权法更新频率：每 N 个样本重新计算一次权重
        self.entropy_update_interval = config.ENTROPY_UPDATE_INTERVAL
        self.min_samples_for_entropy = config.MIN_SAMPLES_FOR_ENTROPY  # 最少样本数才启用熵权法

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

    def compute_ntk_condition_number(self, network: nn.Module, xloader, num_batch: int = -1, train_mode: bool = True) -> float:
        """严格按照 ntk.py 中的 get_ntk_n 逻辑计算单个网络的 NTK 条件数"""
        device = self.device if self.device == 'cpu' else 0

        # ✅ 修复1: 使用 train 模式
        if train_mode:
            network.train()
        else:
            network.eval()
        
        grads = []

        for i, (inputs, _) in enumerate(xloader):
            if num_batch > 0 and i >= num_batch:
                break
            inputs = inputs.to(device=device, non_blocking=True)

            network.zero_grad()
            logit = network(inputs)
            if isinstance(logit, tuple):
                logit = logit[1]

            for idx in range(inputs.size(0)):
                logit[idx:idx + 1].backward(torch.ones_like(logit[idx:idx + 1]), retain_graph=True)

                grad = []
                for name, p in network.named_parameters():
                    if 'weight' in name and p.grad is not None:
                        grad.append(p.grad.view(-1).detach().clone())  # ✅ 需要 clone，防止 zero_grad 清空

                if grad:
                    grads.append(torch.cat(grad, -1))  # ✅ 加上 dim=-1 保持一致

                network.zero_grad()  # ✅ 移到这里，和原始代码一致
                # torch.cuda.empty_cache() # 移出循环以提升速度

        if len(grads) == 0:
            return 100000.0

        grads_tensor = torch.stack(grads, 0)  # (N, C)
        ntk = torch.einsum('nc,mc->nm', [grads_tensor, grads_tensor])  # ✅ 注意括号格式

        # ✅ 修复2: 正确处理特征值
        try:
            eigenvalues = torch.linalg.eigvalsh(ntk)  # 只返回特征值，更高效
        except AttributeError:
            eigenvalues, _ = torch.symeig(ntk)  # ✅ 正确解包

        # 使用绝对值避免负特征值导致的负条件数
        eigenvalues_abs = torch.abs(eigenvalues)
        max_eigen = eigenvalues_abs.max().item()
        min_eigen = eigenvalues_abs.min().item()
        
        # 避免除零
        if min_eigen < 1e-10:
            cond = 100000.0
        else:
            cond = max_eigen / min_eigen
        cond = np.nan_to_num(cond, nan=100000.0, posinf=100000.0, neginf=100000.0)

        del grads, grads_tensor, ntk, eigenvalues
        clear_gpu_memory()

        return cond


    def compute_ntk_score(self, network: nn.Module, param_count: int = None, num_runs: int = 5) -> float:
        """计算 NTK 分数，多次运行取平均"""
        try:
            if param_count and param_count > self.param_threshold:
                logger.warning(f"Skipping NTK: params {param_count} > threshold {self.param_threshold}")
                return 100000.0

            network = network.to(self.device)

            if self.recalbn > 0:
                network = self.recal_bn(network, self.trainloader, self.recalbn, self.device)

            # ✅ 修复3: 多次计算取平均
            total_cond = 0.0
            for _ in range(num_runs):
                cond = self.compute_ntk_condition_number(
                    network, self.trainloader, 
                    num_batch=self.num_batch, 
                    train_mode=True  # ✅ 使用 train 模式
                )
                total_cond += cond
            
            return round(total_cond / num_runs, 3)

        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return 100000.0
    
    def _update_history(self, ntk_score: float, param_count: int):
        """更新历史数据"""
        self.history_ntk.append(ntk_score)
        self.history_param.append(float(param_count))
        
        # 定期更新熵权重
        n = len(self.history_ntk)
        if n >= self.min_samples_for_entropy and n % self.entropy_update_interval == 0:
            self._compute_entropy_weights()
    
    def _compute_entropy_weights(self):
        """
        熵权法计算权重
        
        原理：
        1. 对每个目标进行归一化
        2. 计算每个目标的信息熵 H_j
        3. 熵越小表示该目标区分度越高，应赋予更大权重
        4. 权重 w_j = (1 - H_j) / sum(1 - H_k)
        """
        n = len(self.history_ntk)
        # 至少需要 2 个样本才能计算熵
        min_required = max(2, self.min_samples_for_entropy)
        if n < min_required:
            return
        
        # 1. 获取动态归一化范围（从历史数据中）
        ntk_min, ntk_max = min(self.history_ntk), max(self.history_ntk)
        param_min, param_max = min(self.history_param), max(self.history_param)
        
        # 防止除零
        if ntk_max <= ntk_min:
            ntk_max = ntk_min + 1.0
        if param_max <= param_min:
            param_max = param_min + 1.0
        
        self.dynamic_normalize_range = {
            'ntk': (ntk_min, ntk_max),
            'param': (param_min, param_max)
        }
        
        # 2. 归一化所有历史数据到 [0, 1]
        norm_ntk_list = [(v - ntk_min) / (ntk_max - ntk_min) for v in self.history_ntk]
        norm_param_list = [(v - param_min) / (param_max - param_min) for v in self.history_param]
        
        # 3. 计算每个目标的熵
        def compute_entropy(values: List[float]) -> float:
            """计算信息熵 H = -sum(p * ln(p))"""
            # 避免零值
            eps = 1e-10
            values = [max(v, eps) for v in values]
            total = sum(values)
            if total <= 0:
                return 0.0
            
            # 归一化为概率分布
            probs = [v / total for v in values]
            
            # 计算熵（归一化到 [0, 1]）
            entropy = 0.0
            for p in probs:
                if p > eps:
                    entropy -= p * np.log(p)
            
            # 归一化熵：除以 ln(n) 使其范围为 [0, 1]
            max_entropy = np.log(len(values)) if len(values) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
        
        H_ntk = compute_entropy(norm_ntk_list)
        H_param = compute_entropy(norm_param_list)
        
        # 4. 计算权重：w_j = (1 - H_j) / sum(1 - H_k)
        # 熵越小，区分度越高，权重越大
        d_ntk = 1.0 - H_ntk    # 差异系数
        d_param = 1.0 - H_param
        
        total_d = d_ntk + d_param
        if total_d <= 0:
            # 退化为等权重
            w_ntk, w_param = 0.5, 0.5
        else:
            w_ntk = d_ntk / total_d
            w_param = d_param / total_d
        
        self.entropy_weights = (w_ntk, w_param)
        
        logger.info(f"[EntropyWeight] Updated weights: NTK={w_ntk:.4f}, Param={w_param:.4f} "
                   f"(H_ntk={H_ntk:.4f}, H_param={H_param:.4f}, samples={n})")
    
    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """
        归一化到 [0, 1] 范围
        """
        if max_val <= min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))  # 裁剪到 [0, 1]
    
    def compute_weighted_fitness(self, ntk_score: float, param_count: int) -> float:
        """
        计算加权多目标适应度（熵权法）
        
        1. 样本数不足时使用配置的固定权重
        2. 样本数足够后使用熵权法动态计算的权重
        3. 使用历史数据的动态范围进行归一化
        
        Args:
            ntk_score: NTK 条件数（越小越好）
            param_count: 模型参数量（越小越好）
        
        Returns:
            加权适应度值（越小越好）
        """
        # 更新历史数据
        self._update_history(ntk_score, param_count)
        
        # 确定归一化范围
        if self.dynamic_normalize_range is not None:
            ntk_min, ntk_max = self.dynamic_normalize_range['ntk']
            param_min, param_max = self.dynamic_normalize_range['param']
        else:
            # 使用配置的静态范围
            ntk_min, ntk_max = config.NTK_NORMALIZE_MIN, config.NTK_NORMALIZE_MAX
            param_min, param_max = config.PARAM_NORMALIZE_MIN, config.PARAM_NORMALIZE_MAX
        
        # 归一化
        norm_ntk = self.normalize(ntk_score, ntk_min, ntk_max)
        norm_param = self.normalize(param_count, param_min, param_max)
        
        # 确定权重
        if self.entropy_weights is not None:
            w_ntk, w_param = self.entropy_weights
        else:
            # 使用配置的固定权重
            w_ntk = config.MULTI_OBJ_WEIGHT_NTK
            w_param = config.MULTI_OBJ_WEIGHT_PARAM
            # 归一化权重
            total_w = w_ntk + w_param
            if total_w > 0:
                w_ntk /= total_w
                w_param /= total_w
        
        # 加权求和
        weighted_fitness = w_ntk * norm_ntk + w_param * norm_param
        
        return round(weighted_fitness, 6)

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            network = NetworkBuilder.build_from_individual(
                individual,
                input_channels=self.input_size[0],
                num_classes=self.num_classes
            )
            param_count = network.get_param_count()
            individual.param_count = param_count

            ntk_score = self.compute_ntk_score(network, param_count)
            individual.ntk_score = ntk_score  # 保存原始 NTK 值

            # 计算加权多目标适应度（归一化后加权）
            fitness = self.compute_weighted_fitness(ntk_score, param_count)
            individual.fitness = fitness

            del network
            clear_gpu_memory()

            logger.log_evaluation(individual.id, "MultiObj", fitness, param_count, 
                                  extra_info=f"NTK={ntk_score:.2f}")
            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = 1.0  # 归一化后的最大值
            individual.ntk_score = 100000.0
            clear_gpu_memory()
            return 1.0


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
        elif dataset == 'imagenet':
            self.trainloader, self.testloader = DatasetLoader.get_imagenet()
            self.num_classes = config.IMAGENET_NUM_CLASSES
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def plot_training_history(self, history: list, individual_id: int, epochs: int, 
                               best_acc: float, param_count: int, save_dir: str):
        """
        绘制并保存训练曲线图
        
        Args:
            history: 训练历史记录列表
            individual_id: 个体ID
            epochs: 训练轮数
            best_acc: 最佳准确率
            param_count: 模型参数量
            save_dir: 保存目录
        """
        if not history:
            return
        
        # 提取数据
        epoch_list = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        test_loss = [h['test_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        test_acc = [h['test_acc'] for h in history]
        lr_list = [h['lr'] for h in history]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training History - Model {individual_id}\n'
                     f'Params: {param_count:,} | Best Test Acc: {best_acc:.2f}% | Epochs: {epochs}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Loss 曲线
        ax1 = axes[0, 0]
        ax1.plot(epoch_list, train_loss, 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epoch_list, test_loss, 'r-', label='Test Loss', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Accuracy 曲线
        ax2 = axes[0, 1]
        ax2.plot(epoch_list, train_acc, 'b-', label='Train Acc', linewidth=1.5)
        ax2.plot(epoch_list, test_acc, 'r-', label='Test Acc', linewidth=1.5)
        ax2.axhline(y=best_acc, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Train/Test Acc 差距（过拟合指标）
        ax3 = axes[1, 0]
        gap = [train_acc[i] - test_acc[i] for i in range(len(train_acc))]
        ax3.plot(epoch_list, gap, 'purple', linewidth=1.5)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.fill_between(epoch_list, 0, gap, alpha=0.3, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train Acc - Test Acc (%)')
        ax3.set_title('Generalization Gap (Overfitting Indicator)')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        ax4.plot(epoch_list, lr_list, 'green', linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f'training_curve_model_{individual_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Training curve saved to {plot_path}")
        return plot_path

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

        # 保存模型
        save_dir = os.path.join(config.CHECKPOINT_DIR, 'final_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{individual.id}_acc{best_acc:.2f}.pth')

        save_dict = {
            'state_dict': network.state_dict(),
            'encoding': individual.encoding,
            'accuracy': best_acc,
            'param_count': param_count,
            'fitness': individual.fitness,
            'ntk_score': individual.ntk_score,
            'history': history
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved model to {save_path}")
        logger.info(f"Model {individual.id} Architecture Encoding: {individual.encoding}")
        
        # 生成并保存训练曲线图到 logs 目录
        plot_dir = os.path.join(config.LOG_DIR, 'training_curves')
        try:
            plot_path = self.plot_training_history(
                history=history,
                individual_id=individual.id,
                epochs=epochs,
                best_acc=best_acc,
                param_count=param_count,
                save_dir=plot_dir
            )
        except Exception as e:
            logger.warning(f"Failed to generate training curve plot: {e}")
            plot_path = None

        result = {
            'individual_id': individual.id,
            'param_count': param_count,
            'best_accuracy': best_acc,
            'fitness': individual.fitness,
            'ntk_score': individual.ntk_score,
            'train_time': train_time,
            'history': history,
            'encoding': individual.encoding,
            'model_path': save_path,
            'plot_path': plot_path
        }
        return best_acc, result

    def evaluate_top_individuals(self, population: List[Individual],
                                top_k: int = None, epochs: int = None) -> Tuple[Individual, List[dict]]:
        if top_k is None:
            top_k = config.HISTORY_TOP_N2
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS

        sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness is not None else float('inf'), reverse=False)
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
    """
    适应度评估器
    延迟加载 NTKEvaluator，避免模块导入时加载数据集
    """
    def __init__(self):
        self._ntk_evaluator = None

    @property
    def ntk_evaluator(self):
        """延迟加载 NTKEvaluator，使用当前 config 配置"""
        if self._ntk_evaluator is None:
            self._ntk_evaluator = NTKEvaluator(dataset=config.FINAL_DATASET)
        return self._ntk_evaluator

    def reset(self):
        """重置评估器，用于切换数据集后重新初始化"""
        self._ntk_evaluator = None

    def evaluate_individual(self, individual: Individual) -> float:
        """评估单个个体的 NTK Fitness"""
        return self.ntk_evaluator.evaluate_individual(individual)


# 全局实例
fitness_evaluator = FitnessEvaluator()