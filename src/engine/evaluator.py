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

    def compute_ntk_condition_number(
        self,
        network: nn.Module,
        xloader,
        num_batch: int = 1,
        train_mode: bool = False,
    ) -> float:
        """
        Compute NTK condition number using loop-based gradient accumulation.

        Args:
            network (nn.Module): The neural network.
            xloader: Data loader.
            num_batch (int): Limit number of batches.
            train_mode (bool): Whether to set model to train mode.

        Returns:
            float: The NTK condition number.
        """
        device = torch.cuda.current_device() if self.device == 'cuda' else 'cpu'
        networks = [network]
        
        for net in networks:
            if train_mode:
                net.train()
            else:
                net.eval()

        grads = [[] for _ in range(len(networks))]
        for i, (inputs, targets) in enumerate(xloader):
            if num_batch > 0 and i >= num_batch:
                break
            inputs = inputs.to(device=device, non_blocking=True)
            for net_idx, net in enumerate(networks):
                net.zero_grad()
                inputs_ = inputs.clone().to(device=device, non_blocking=True)
                logit = net(inputs_)
                if isinstance(logit, tuple):
                    logit = logit[1]
                for _idx in range(len(inputs_)):
                    logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                    grad = []
                    for name, W in net.named_parameters():
                        if 'weight' in name and W.grad is not None:
                            grad.append(W.grad.view(-1).detach())
                    grads[net_idx].append(torch.cat(grad, -1))
                    net.zero_grad()
                    torch.cuda.empty_cache()

        grads = [torch.stack(_grads, 0) for _grads in grads]
        ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
        conds = []
        for ntk in ntks:
            try:
                
                # Use robust eigenvalue calculation
                eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U')  # ascending
                
                # Enforce PSD (Positive Semi-Definite) property by clamping small/negative values to a small epsilon
                # 1e-6 was too large for some networks with small gradients (e.g. ~1e-12), causing NTK=1.0
                eigenvalues = torch.clamp(eigenvalues, min=1e-30)
                
                # Check for NaNs
                if torch.isnan(eigenvalues).any():
                     logger.warning("NaN detected in eigenvalues")
                     conds.append(100000.0)
                     continue
                
                # Check for Dead Network (Vanishing Gradients)
                # If the maximum eigenvalue is extremely small, the network has no gradients.
                # In this case, min~max~1e-30, resulting in cond=1.0, which is misleadingly "perfect".
                if eigenvalues[-1].item() < 1e-8:
                     logger.warning(f"Dead network detected (Max Eigenvalue < 1e-8: {eigenvalues[-1].item():.6e}). Penalizing.")
                     conds.append(100000.0)
                     continue

                condition_number = (eigenvalues[-1] / eigenvalues[0]).item()
                conds.append(np.nan_to_num(condition_number, copy=True, nan=100000.0))
            except Exception as e:
                logger.warning(f"NTK eigenvalue computation failed: {e}")
                conds.append(100000.0)
                
        return conds[0]

    def compute_ntk_score(self, network: nn.Module, param_count: int = None, num_runs: int = 1) -> float:
        """
        Compute NTK score by averaging multiple runs (removing min/max if num_runs > 2).
        
        Args:
            network (nn.Module): The model.
            param_count (int, optional): Parameter count for threshold check.
            num_runs (int): Number of independent NTK calculations to average.

        Returns:
            float: Averaged NTK condition number.
        """
        try:
            if param_count and param_count > self.param_threshold:
                logger.warning(f"Skipping NTK: params {param_count} > threshold {self.param_threshold}")
                return 100000.0

            network = network.to(self.device)

            if self.recalbn > 0:
                network = self.recal_bn(network, self.trainloader, self.recalbn, self.device)

            cond_scores = []
            for _ in range(num_runs):
                score = self.compute_ntk_condition_number(
                    network,
                    self.trainloader,
                    num_batch=self.num_batch,
                    train_mode=False,
                )
                cond_scores.append(score)

            # 如果运行次数 > 2，移除最大和最小值再求平均
            if len(cond_scores) > 2:
                cond_scores.remove(max(cond_scores))
                cond_scores.remove(min(cond_scores))
            
            avg_score = sum(cond_scores) / len(cond_scores)
            return round(avg_score, 3)

        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return 100000.0
    def evaluate_individual(self, individual: Individual) -> float:
        try:
            network = NetworkBuilder.build_from_individual(
                individual,
                num_classes=self.num_classes,
                cells_per_stage=config.NTK_CELLS_PER_STAGE,  # 使用较浅的网络进行NTK评估
                enable_dropout=False  # NTK 评估时禁用 Dropout
            )
            param_count = network.get_param_count()
            individual.param_count = param_count

            score = self.compute_ntk_score(network, param_count)

            # fitness 直接等于 NTK 条件数（越小越好）
            fitness = score
            individual.fitness = fitness

            del network
            clear_gpu_memory()

            logger.log_evaluation(individual.id, "NTK", fitness, param_count)
            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = 100000.0  # 极大值表示最差
            clear_gpu_memory()
            return 100000.0


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

    def evaluate_individual(self, individual: Individual, epochs: int = None, model_type: str = 'full') -> Tuple[float, dict]:
        """
        训练并评估个体
        
        Args:
            individual: 待评估的个体
            epochs: 训练轮数
            model_type: 模型类型，'short' 表示短期训练，'full' 表示完整训练
        """
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS

        logger.info(f"Training individual {individual.id} for {epochs} epochs...")
        network = NetworkBuilder.build_from_individual(
            individual, num_classes=self.num_classes
        )
        param_count = network.get_param_count()
        individual.param_count = param_count
        Encoder.print_architecture(individual)

        start_time = time.time()
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader, epochs
        )
        train_time = time.time() - start_time

        # 根据模型类型选择保存目录
        if model_type == 'short':
            save_dir = os.path.join(config.CHECKPOINT_DIR, 'short_train_models')
        else:
            save_dir = os.path.join(config.CHECKPOINT_DIR, 'full_train_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{individual.id}_acc{best_acc:.2f}.pth')

        # 保存genotype格式
        genotype = Encoder.get_genotype(individual)
        save_dict = {
            'state_dict': network.state_dict(),
            'genotype': genotype,
            'normal_cell': individual.normal_cell.to_list(),
            'reduction_cell': individual.reduction_cell.to_list(),
            'accuracy': best_acc,
            'param_count': param_count,
            'history': history
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved model to {save_path}")
        logger.info(f"Model {individual.id} Genotype: {genotype}")
        
        # 根据模型类型分开存储训练曲线图
        if model_type == 'short':
            plot_dir = os.path.join(config.LOG_DIR, 'training_curves', 'short_train')
        else:
            plot_dir = os.path.join(config.LOG_DIR, 'training_curves', 'full_train')
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
            'train_time': train_time,
            'history': history,
            'genotype': genotype,
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
            logger.info(f"[{idx + 1}/{top_k}] Evaluating Individual {individual.id}")
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