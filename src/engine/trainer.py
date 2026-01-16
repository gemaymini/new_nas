# -*- coding: utf-8 -*-
"""
Training utilities for searched networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import Tuple, List
from configuration.config import config
from utils.logger import logger


class NetworkTrainer:
    """Train and evaluate a network."""

    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU for training")

    def _get_param_groups(self, model: nn.Module, weight_decay: float):
        """Separate parameters to avoid decaying biases/norm layers."""
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _build_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        betas: tuple,
        eps: float,
        momentum: float = None,
        nesterov: bool = None,
    ) -> optim.Optimizer:
        """Create optimizer based on config choice."""
        name = optimizer_name.lower()
        params = self._get_param_groups(model, weight_decay)

        if name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        if name == "sgd":
            return optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum if momentum is not None else config.SGD_MOMENTUM,
                nesterov=nesterov if nesterov is not None else config.SGD_NESTEROV,
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train_one_epoch(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Tuple[float, float]:
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
                acc = 100.0 * correct / total
                print(
                    f"\rPROGRESS: Train {epoch}/{total_epochs} "
                    f"batch {batch_idx + 1}/{len(trainloader)} "
                    f"({progress:.1f}%) loss={running_loss/(batch_idx+1):.4f} acc={acc:.2f}%",
                    end="",
                    flush=True,
                )
        print()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def evaluate(
        self,
        model: nn.Module,
        testloader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
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
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train_network(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int = None,
        lr: float = None,
        weight_decay: float = None,
        betas: tuple = None,
        eps: float = None,
        early_stopping: bool = None,
        optimizer_name: str = None,
        warmup_epochs: int = None,
    ) -> Tuple[float, List[dict]]:
        if epochs is None:
            epochs = config.FULL_TRAIN_EPOCHS
        if optimizer_name is None:
            optimizer_name = config.OPTIMIZER
        optimizer_name = optimizer_name.lower()
        optimizer_defaults = config.get_optimizer_params(optimizer_name)

        if lr is None:
            lr = optimizer_defaults["lr"]
        if weight_decay is None:
            weight_decay = optimizer_defaults["weight_decay"]
        if betas is None:
            betas = optimizer_defaults.get("betas", None)
        if eps is None:
            eps = optimizer_defaults.get("eps", config.ADAMW_EPS)
        if early_stopping is None:
            early_stopping = config.EARLY_STOPPING_ENABLED
        if warmup_epochs is None:
            warmup_epochs = optimizer_defaults.get("warmup_epochs", 0)

        momentum = optimizer_defaults.get("momentum", None)
        nesterov = optimizer_defaults.get("nesterov", None)
        eta_min = lr * getattr(config, "COSINE_MIN_LR_RATIO", 0.0)

        patience = config.EARLY_STOPPING_PATIENCE
        min_delta = config.EARLY_STOPPING_MIN_DELTA
        patience_counter = 0

        if self.device == "cuda":
            torch.cuda.empty_cache()

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(
            model,
            optimizer_name,
            lr,
            weight_decay,
            betas,
            eps,
            momentum=momentum,
            nesterov=nesterov,
        )

        if warmup_epochs > 0 and warmup_epochs < epochs:
            warmup = lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: float(epoch + 1) / float(warmup_epochs)
            )
            cosine = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min
            )
            scheduler = lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs), eta_min=eta_min
            )

        hyper_parts = [
            f"optimizer={optimizer_name}",
            f"lr={lr}",
            f"weight_decay={weight_decay}",
        ]
        if optimizer_name == "adamw":
            hyper_parts.append(f"betas={betas}")
            hyper_parts.append(f"eps={eps}")
        elif optimizer_name == "sgd":
            hyper_parts.append(f"momentum={momentum}")
            hyper_parts.append(f"nesterov={nesterov}")
        hyper_parts.append(f"warmup_epochs={warmup_epochs}")
        hyper_parts.append(f"eta_min={eta_min}")
        logger.info("Optimizer setup: " + " ".join(hyper_parts))

        history = []
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        try:
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = self.train_one_epoch(
                    model, trainloader, criterion, optimizer, epoch, epochs
                )
                test_loss, test_acc = self.evaluate(model, testloader, criterion)
                scheduler.step()

                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "optimizer": optimizer_name,
                        "warmup_epochs": warmup_epochs,
                    }
                )

                if test_acc > best_acc + min_delta:
                    best_acc = test_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                es_info = f" | ES: {patience_counter}/{patience}" if early_stopping else ""
                print(
                    f"INFO: epoch {epoch}/{epochs} train_acc={train_acc:.2f}% "
                    f"test_acc={test_acc:.2f}% best={best_acc:.2f}%{es_info}"
                )

                if early_stopping and patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best accuracy: {best_acc:.2f}%"
                    )
                    break

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("OOM during training. Clearing cache.")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            raise e
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

        model.load_state_dict(best_model_wts)
        return best_acc, history
