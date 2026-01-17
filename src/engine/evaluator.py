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
from models.network import NetworkBuilder
from utils.logger import logger
from data.dataset import DatasetLoader
from engine.trainer import NetworkTrainer



def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()



def _allocate_budget(lengths, budget, min_per=0):
    """
    lengths: list[int] each tensor length
    returns alloc list[int] sum == budget (or == sum(lengths) if budget larger)
    """
    total = sum(lengths)
    if budget >= total:
        return lengths[:]  # take all

    # min allocation
    alloc = [min(l, min_per) for l in lengths]
    s = sum(alloc)
    if s > budget:
        # shrink mins proportionally
        # start from zeros and distribute by lengths
        alloc = [0] * len(lengths)
        # proportional base
        frac = [l / total for l in lengths]
        alloc = [min(lengths[i], int(frac[i] * budget)) for i in range(len(lengths))]
        diff = budget - sum(alloc)
        # distribute remaining to largest remaining capacity
        cap = [lengths[i] - alloc[i] for i in range(len(lengths))]
        order = sorted(range(len(lengths)), key=lambda i: cap[i], reverse=True)
        for i in order:
            if diff == 0: break
            if cap[i] > 0:
                alloc[i] += 1
                cap[i] -= 1
                diff -= 1
        return alloc

    remaining = budget - s
    # distribute remaining proportionally to (length - alloc)
    rem_caps = [lengths[i] - alloc[i] for i in range(len(lengths))]
    rem_total = sum(rem_caps)
    if rem_total <= 0:
        return alloc

    # proportional extras
    extra = [int(rem_caps[i] / rem_total * remaining) for i in range(len(lengths))]
    # cap extras
    extra = [min(extra[i], rem_caps[i]) for i in range(len(lengths))]
    diff = remaining - sum(extra)

    # distribute leftover to largest remaining capacity
    cap2 = [rem_caps[i] - extra[i] for i in range(len(lengths))]
    order = sorted(range(len(lengths)), key=lambda i: cap2[i], reverse=True)
    for i in order:
        if diff == 0: break
        if cap2[i] > 0:
            extra[i] += 1
            cap2[i] -= 1
            diff -= 1

    return [alloc[i] + extra[i] for i in range(len(lengths))]


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
                m.running_var.data.fill_(1)  # Initialize with 1 to avoid initial instability
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
        num_batch: int = 1,
        train_mode: bool = False,
    ) -> float:
        """
        Compute NTK condition number using vectorized gradients (torch.func).

        Args:
            network (nn.Module): The neural network.
            xloader: Data loader.
            num_batch (int): Limit number of batches.
            train_mode (bool): Whether to set model to train mode.

        Returns:
            float: The NTK condition number (log10 scale, clipped).
        """
        from torch.func import functional_call, vmap, grad
        
        device = torch.device(self.device)
        network.train() if train_mode else network.eval()

        # Config
        ntk_target = getattr(config, "NTK_TARGET", "sum_logits")
        ntk_max_params = int(getattr(config, "NTK_MAX_PARAMS", 200_000))
        ntk_cond_clip = float(getattr(config, "NTK_COND_CLIP", 1e8))
        ntk_eig_dtype = getattr(config, "NTK_EIG_DTYPE", "float64")
        max_score = float(np.log10(ntk_cond_clip + 1.0))

        # Prepare parameters
        params = {k: v for k, v in network.named_parameters() if v.requires_grad}
        param_names = list(params.keys())

        grads = []
        slices = None

        for i, (inputs, targets) in enumerate(xloader):
            if num_batch > 0 and i >= num_batch:
                break
            inputs = inputs.to(device=device, non_blocking=True)
            if ntk_target == "true_logit":
                targets = targets.to(device=device, non_blocking=True)

            # Define loss function for gradient computation
            def compute_loss(params, x, y=None):
                outputs = functional_call(network, params, (x.unsqueeze(0),)).squeeze(0)
                if ntk_target == "true_logit" and y is not None:
                    return outputs.gather(0, y.view(1)).sum()
                return outputs.sum()

            # Compute per-sample gradients via vmap
            if ntk_target == "true_logit":
                batch_grads_dict = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, inputs, targets)
            else:
                batch_grads_dict = vmap(grad(compute_loss), in_dims=(None, 0, None))(params, inputs, None)
            
            # Build slices on first batch (guaranteed alignment)
            if slices is None:
                slices = []
                offset = 0
                for k in param_names:
                    length = batch_grads_dict[k][0].numel()
                    slices.append((k, offset, offset + length))
                    offset += length

            # Flatten and concatenate gradients
            batch_grads = torch.cat([batch_grads_dict[k].flatten(start_dim=1) for k in param_names], dim=1)
            grads.append(batch_grads)

        if len(grads) == 0:
            return max_score

        grads_tensor = torch.cat(grads, dim=0)  # (TotalSamples, TotalParams)
        
        # Layer-wise subsampling
        P = grads_tensor.shape[1]
        if P > ntk_max_params:
            min_per_tensor = int(getattr(config, "NTK_MIN_PER_TENSOR", 16))
            seed = int(getattr(config, "NTK_SUBSAMPLE_SEED", 42))

            lengths = [end - start for _, start, end in slices]
            alloc = _allocate_budget(lengths, ntk_max_params, min_per=min_per_tensor)

            g = torch.Generator(device=grads_tensor.device)
            
            g.manual_seed(seed)

            idx_list = []
            for (_, start, end), k in zip(slices, alloc):
                length = end - start
                if k <= 0:
                    continue
                if k >= length:
                    local = torch.arange(length, device=grads_tensor.device, dtype=torch.long)
                else:
                    local = torch.randperm(length, generator=g, device=grads_tensor.device)[:k]
                idx_list.append(local + start)

            if idx_list:
                grads_tensor = grads_tensor[:, torch.cat(idx_list, dim=0)]

        # Compute NTK (Gram Matrix)
        ntk = grads_tensor @ grads_tensor.t()

        # Eigenvalues
        if ntk_eig_dtype == "float64":
            eigenvalues = torch.linalg.eigvalsh(ntk.double())
        else:
            eigenvalues = torch.linalg.eigvalsh(ntk)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        
        max_eigen = eigenvalues.max().item()
        min_eigen = eigenvalues.min().item()

        cond = ntk_cond_clip if min_eigen < 1e-12 else max_eigen / min_eigen
        cond = min(cond, ntk_cond_clip)
        score = float(np.log10(cond + 1.0))
        
        logger.info(f"NTK Linear: {cond:.4e}, LogScore: {score:.4f}")

        del grads, grads_tensor, ntk, eigenvalues
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return score

    def compute_ntk_score(self, network: nn.Module, num_runs: int = 3) -> float:
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
            # Return max clipped score on failure
            ntk_cond_clip = float(getattr(config, "NTK_COND_CLIP", 1e8))
            return float(np.log10(ntk_cond_clip + 1.0))

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
            # Return max clipped score on failure
            ntk_cond_clip = float(getattr(config, "NTK_COND_CLIP", 1e8))
            max_score = float(np.log10(ntk_cond_clip + 1.0))
            individual.fitness = max_score
            clear_gpu_memory()
            return max_score


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
