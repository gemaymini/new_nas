# -*- coding: utf-8 -*-
"""
网络训练器模块
负责网络的训练和评估循环
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from typing import Tuple, List
from utils.config import config
from utils.logger import logger

class NetworkTrainer:
    """
    网络训练器
    """
    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logger.warning("CUDA not available, using CPU for training")
            
    def train_one_epoch(self, model: nn.Module, trainloader: DataLoader,
                       criterion: nn.Module, optimizer: optim.Optimizer,
                       epoch: int, total_epochs: int) -> Tuple[float, float]:
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
        if epochs is None: epochs = config.FINAL_TRAIN_EPOCHS
        if lr is None: lr = config.LEARNING_RATE
        if momentum is None: momentum = config.MOMENTUM
        if weight_decay is None: weight_decay = config.WEIGHT_DECAY
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, 
                             momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = []
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        
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
                best_model_wts = copy.deepcopy(model.state_dict())
            
            print(f'  [Epoch {epoch}/{epochs}] Train Acc: {train_acc:.2f}% | '
                  f'Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%')
            
        model.load_state_dict(best_model_wts)
        return best_acc, history
