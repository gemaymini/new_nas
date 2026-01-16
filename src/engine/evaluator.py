# -*- coding: utf-8 -*-
"""
Evaluation utilities for NTK scoring and full training.
"""
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
from core.search_space import population_initializer
from models.network import NetworkBuilder
from utils.logger import logger
from data.dataset import DatasetLoader
from engine.trainer import NetworkTrainer



def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class NTKEvaluator:
    """
    Evaluator using Neural Tangent Kernel (NTK) condition number as a zero-cost proxy.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = None,
        num_classes: int = None,
        batch_size: int = None,
        device: str = None,
        recalbn: int = 0,
        num_batch: int = 1,
        dataset: str = None,
    ):
        """
        Initialize the NTK Evaluator.

        Args:
            input_size (Tuple[int, int, int], optional): Input dimensions (channels, H, W).
            num_classes (int, optional): Number of output classes.
            batch_size (int, optional): Batch size for NTK computation.
            device (str, optional): Computation device ('cuda' or 'cpu').
            recalbn (int, optional): Number of batches for BatchNorm recalibration.
            num_batch (int, optional): Number of batches to use for NTK calculation.
            dataset (str, optional): Dataset name.
        """
        self.input_size = input_size or config.NTK_INPUT_SIZE
        self.num_classes = num_classes or config.NTK_NUM_CLASSES
        self.batch_size = batch_size or config.NTK_BATCH_SIZE
        self.device = device or config.DEVICE
        self.dataset = dataset or config.FINAL_DATASET

        self.recalbn = recalbn if recalbn is not None else config.NTK_RECALBN
        self.num_batch = num_batch if num_batch is not None else config.NTK_NUM_BATCH

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.trainloader = DatasetLoader.get_ntk_trainloader(
            batch_size=self.batch_size,
            dataset_name=self.dataset,
        )

    def recal_bn(self, network: nn.Module, xloader, recal_batches: int, device):
        """
        Reset BN stats and recompute over a few batches.

        Args:
            network (nn.Module): The model to recalibrate.
            xloader: Data loader.
            recal_batches (int): Number of batches to use.
            device: Torch device.
        """
        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.data.fill_(0)
                m.running_var.data.fill_(0)
                m.num_batches_tracked.data.zero_()
                m.momentum = None

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
        num_batch: int = -1,
        train_mode: bool = True,
    ) -> float:
        """
        Compute NTK condition number for a single network using vectorized gradients.

        Args:
            network (nn.Module): The neural network.
            xloader: Data loader.
            num_batch (int): Limit number of batches (-1 for default from init).
            train_mode (bool): Whether to set model to train mode.

        Returns:
            float: The NTK condition number (lower is generally better).
        """
        device = torch.device(self.device)
        if train_mode:
            network.train()
        else:
            network.eval()

        # Try to use torch.func for vectorized computation (PyTorch 2.0+)
        try:
            from torch.func import functional_call, vmap, grad
            HAS_TORCH_FUNC = True
        except ImportError:
            HAS_TORCH_FUNC = False
            logger.warning("torch.func not found; falling back to slow loop-based NTK calculation.")

        grads = []
        
        # Prepare parameters for functional call if available
        if HAS_TORCH_FUNC:
            params = {k: v for k, v in network.named_parameters() if v.requires_grad}
            
            def fnet_single(params, x):
                output = functional_call(network, params, (x.unsqueeze(0),))
                return output.squeeze(0)

        for i, (inputs, _) in enumerate(xloader):
            if num_batch > 0 and i >= num_batch:
                break
            inputs = inputs.to(device=device, non_blocking=True)

            if HAS_TORCH_FUNC:
                # Vectorized computation
                # Compute gradients for the whole batch at once: (Batch, Params)
                # functional_call expects a dict of params. 
                # We need to trace gradients w.r.t these params.
                
                # We need a wrapper that returns the sum of logits or specific logit to differentiate
                # But classical NTK for classification usually takes gradients of the output logit corresponding to the class,
                # or simplified: gradients of the scalar output sum (if regression) or similar.
                # Here, the original code did: logit[idx:idx+1].backward(torch.ones_like(...))
                # This implies scalar sum of outputs for that sample (since it's a vector output, backward with ones sums gradients).
                


                # Correct functional approach for per-sample gradients:
                # 1. Define function: params, single_input -> scalar_loss
                def compute_loss(params, x):
                    outputs = functional_call(network, params, (x.unsqueeze(0),))
                    return outputs.sum() # Sum all logits as proxy target

                # 2. Vectorize grad over the batch of inputs
                batch_grads_dict = vmap(grad(compute_loss), in_dims=(None, 0))(params, inputs)
                
                # 3. Flatten and concatenate all parameter gradients for each sample
                # batch_grads_dict is {param_name: tensor(Batch, ...)}
                # We want (Batch, TotalParams)
                batch_grads_list = [batch_grads_dict[k].flatten(start_dim=1) for k in batch_grads_dict]
                batch_grads = torch.cat(batch_grads_list, dim=1) # (Batch, TotalParams)
                grads.append(batch_grads)

            else:
                # Fallback: Slow loop
                network.zero_grad()
                logit = network(inputs)
                if isinstance(logit, tuple):
                    logit = logit[1]

                for idx in range(inputs.size(0)):
                    logit[idx:idx + 1].backward(
                        torch.ones_like(logit[idx:idx + 1]),
                        retain_graph=True,
                    )

                    grad_list = []
                    for name, p in network.named_parameters():
                        # Include all trainable parameters (weights + biases)
                        if p.grad is not None and p.requires_grad:
                            grad_list.append(p.grad.view(-1).detach().clone())
                    
                    if grad_list:
                        grads.append(torch.cat(grad_list, -1))

                    network.zero_grad()

        if len(grads) == 0:
            return 10000000000.0

        if HAS_TORCH_FUNC:
             grads_tensor = torch.cat(grads, dim=0) # (TotalBatch, TotalParams)
        else:
             grads_tensor = torch.stack(grads, 0)

        # Standardize assumption: If grads are too large, we might OOM on dot product.
        # But for NTK proxy runs (small batch, few batches), it's usually fine.
        
        ntk = torch.einsum("nc,mc->nm", [grads_tensor, grads_tensor])

        try:
            eigenvalues = torch.linalg.eigvalsh(ntk)
        except AttributeError:
            eigenvalues, _ = torch.symeig(ntk)

        eigenvalues_abs = torch.abs(eigenvalues)
        max_eigen = eigenvalues_abs.max().item()
        min_eigen = eigenvalues_abs.min().item()

        if min_eigen < 1e-10:
            cond = 10000000000.0
        else:
            cond = max_eigen / min_eigen
        cond = np.nan_to_num(cond, nan=10000000000.0, posinf=10000000000.0, neginf=10000000000.0)

        del grads, grads_tensor, ntk, eigenvalues
        clear_gpu_memory()

        return cond

    def compute_ntk_score(self, network: nn.Module, num_runs: int = 5) -> float:
        """
        Compute NTK score by averaging multiple runs.
        
        Args:
            network (nn.Module): The model.
            num_runs (int): Number of independent NTK calculations to average.

        Returns:
            float: Averaged NTK condition number.
        """
        try:
            network = network.to(self.device)

            if self.recalbn > 0:
                network = self.recal_bn(network, self.trainloader, self.recalbn, self.device)

            total_cond = 0.0
            for _ in range(num_runs):
                cond = self.compute_ntk_condition_number(
                    network,
                    self.trainloader,
                    num_batch=self.num_batch,
                    train_mode=False,
                )
                total_cond += cond

            return round(total_cond / num_runs, 3)

        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return 10000000000.0

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            network = NetworkBuilder.build_from_individual(
                individual,
                input_channels=self.input_size[0],
                num_classes=self.num_classes,
            )
            
            individual.fitness = self.compute_ntk_score(network)

            del network
            clear_gpu_memory()

            logger.log_evaluation(individual.id, "NTK", individual.fitness, individual.param_count)
            return individual.fitness

        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.fitness = 10000000000.0
            clear_gpu_memory()
            return 10000000000.0


class FinalEvaluator:
    """
    Evaluator for final training (screening and full training).
    """
    def __init__(self, dataset: str = "cifar10", device: str = None):
        self.dataset = dataset
        self.device = device or config.DEVICE
        self.trainer = NetworkTrainer(self.device)

        if dataset == "cifar10":
            self.trainloader, self.testloader = DatasetLoader.get_cifar10()
            self.num_classes = 10
        elif dataset == "cifar100":
            self.trainloader, self.testloader = DatasetLoader.get_cifar100()
            self.num_classes = 100
        elif dataset == "imagenet":
            self.trainloader, self.testloader = DatasetLoader.get_imagenet()
            self.num_classes = config.IMAGENET_NUM_CLASSES
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def plot_training_history(
        self,
        history: list,
        individual_id: int,
        epochs: int,
        best_acc: float,
        param_count: int,
        save_dir: str,
    ):
        """
        Plot and save training curves for loss and accuracy.

        Args:
            history (list): List of dicts containing epoch stats.
            individual_id (int): ID of the individual.
            epochs (int): Total epochs.
            best_acc (float): Best test accuracy achieved.
            param_count (int): Number of parameters.
            save_dir (str): Directory to save plots.
        
        Returns:
            str: Path to the saved plot.
        """
        if not history:
            return

        epoch_list = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        test_loss = [h["test_loss"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        test_acc = [h["test_acc"] for h in history]
        lr_list = [h["lr"] for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Training History - Model {individual_id}\n"
            f"Params: {param_count:,} | Best Test Acc: {best_acc:.2f}% | Epochs: {epochs}",
            fontsize=14,
            fontweight="bold",
        )

        ax1 = axes[0, 0]
        ax1.plot(epoch_list, train_loss, "b-", label="Train Loss", linewidth=1.5)
        ax1.plot(epoch_list, test_loss, "r-", label="Test Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = axes[0, 1]
        ax2.plot(epoch_list, train_acc, "b-", label="Train Acc", linewidth=1.5)
        ax2.plot(epoch_list, test_acc, "r-", label="Test Acc", linewidth=1.5)
        ax2.axhline(
            y=best_acc, color="g", linestyle="--", alpha=0.7, label=f"Best: {best_acc:.2f}%"
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curves")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax3 = axes[1, 0]
        gap = [train_acc[i] - test_acc[i] for i in range(len(train_acc))]
        ax3.plot(epoch_list, gap, "purple", linewidth=1.5)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax3.fill_between(epoch_list, 0, gap, alpha=0.3, color="purple")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Train Acc - Test Acc (%)")
        ax3.set_title("Generalization Gap")
        ax3.grid(True, linestyle="--", alpha=0.5)

        ax4 = axes[1, 1]
        ax4.plot(epoch_list, lr_list, "green", linewidth=1.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.set_yscale("log")

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"training_curve_model_{individual_id}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Training curve saved to {plot_path}")
        return plot_path

    def evaluate_individual(self, individual: Individual, epochs: int = None) -> Tuple[float, dict]:
        """
        Train and evaluate an individual.

        Args:
            individual (Individual): The architecture to train.
            epochs (int, optional): Number of training epochs.

        Returns:
            Tuple[float, dict]: Best accuracy and a detailed result dictionary.
        """
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS
        is_short_train = epochs <= config.SHORT_TRAIN_EPOCHS

        logger.info(f"Training individual {individual.id} for {epochs} epochs...")
        network = NetworkBuilder.build_from_individual(
            individual, input_channels=3, num_classes=self.num_classes
        )
        param_count = individual.param_count
        Encoder.print_architecture(individual.encoding)

        start_time = time.time()
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader, epochs
        )
        train_time = time.time() - start_time

        model_dir = config.SHORT_TRAIN_MODEL_DIR if is_short_train else config.FULL_TRAIN_MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"model_{individual.id}_acc{best_acc:.2f}.pth")

        save_dict = {
            "state_dict": network.state_dict(),
            "encoding": individual.encoding,
            "accuracy": best_acc,
            "param_count": param_count,
            "history": history,
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved model to {save_path}")
        logger.info(f"Model {individual.id} Architecture Encoding: {individual.encoding}")

        plot_root = model_dir
        if is_short_train:
            plot_dir = os.path.join(plot_root, "short_training_curves")
        else:
            plot_dir = os.path.join(plot_root, "full_training_curves")
        try:
            plot_path = self.plot_training_history(
                history=history,
                individual_id=individual.id,
                epochs=epochs,
                best_acc=best_acc,
                param_count=param_count,
                save_dir=plot_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to generate training curve plot: {e}")
            plot_path = None

        result = {
            "individual_id": individual.id,
            "param_count": param_count,
            "best_accuracy": best_acc,
            "train_time": train_time,
            "history": history,
            "encoding": individual.encoding,
            "model_path": save_path,
            "plot_path": plot_path,
        }
        return best_acc, result




class FitnessEvaluator:
    def __init__(self):
        self._ntk_evaluator = None

    @property
    def ntk_evaluator(self):
        """Lazy-init NTKEvaluator with current config."""
        if self._ntk_evaluator is None:
            self._ntk_evaluator = NTKEvaluator()
        return self._ntk_evaluator

    def reset(self):
        """Reset evaluator state after dataset changes."""
        self._ntk_evaluator = None

    def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate NTK fitness for a single individual."""
        # Trust upstream validation.
        return self.ntk_evaluator.evaluate_individual(individual)




fitness_evaluator = FitnessEvaluator()
