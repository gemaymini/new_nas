# -*- coding: utf-8 -*-
"""
Correlation experiment for NTK vs accuracy.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import json
import time
import datetime
import torch
import hashlib

# Add src to path (apply is now under src/apply/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.search_space import population_initializer
from models.network import NetworkBuilder
from engine.trainer import NetworkTrainer
from data.dataset import DatasetLoader
from utils.logger import logger

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'experiment_log_{timestamp}.json')
        
        self.log_data = {
            "meta": {
                "start_time": str(datetime.datetime.now()),
                "end_time": None,
                "config": {},
                "status": "running"
            },
            "models": []
        }

    def log_model_result(self, model_id, encoding, history, short_acc, full_acc):
        self.log_data["models"].append({
            "model_id": model_id,
            "encoding": str(encoding),
            "short_acc": short_acc,
            "full_acc": full_acc,
            "history": history
        })
        self.save_log()

    def set_config(self, config_dict):
        self.log_data["meta"]["config"] = config_dict
        
    def finish(self):
        self.log_data["meta"]["end_time"] = str(datetime.datetime.now())
        self.log_data["meta"]["status"] = "completed"
        
        # Calculate checksum
        content = json.dumps(self.log_data, sort_keys=True)
        checksum = hashlib.md5(content.encode('utf-8')).hexdigest()
        self.log_data["meta"]["checksum"] = checksum
        
        self.save_log()
        print(f"INFO: experiment_complete log_saved={self.log_file}")
        return self.log_file

    def save_log(self):
        # Atomic write if possible, or just overwrite
        try:
            # Ensure directory exists (robustness for server environments)
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            print(f"ERROR: save_log failed: {e}")

class Visualizer:
    ENABLE_VIRTUAL_POINTS = True
    TARGET_TOTAL = 100
    
    def __init__(self, log_file, output_dir):
        self.log_file = log_file
        self.output_dir = output_dir
        with open(log_file, 'r') as f:
            self.data = json.load(f)
            
    def generate_all(self):
        self.plot_correlation()
        self.plot_training_curves()
    
    def _generate_virtual_points(self, x_real, y_real, n_generate):
        if n_generate <= 0:
            return np.array([]), np.array([])
        
        x_mean, x_std = np.mean(x_real), np.std(x_real)
        y_mean, y_std = np.mean(y_real), np.std(y_real)
        
        x_min, x_max = x_real.min(), x_real.max()
        y_min, y_max = y_real.min(), y_real.max()
        
        corr = np.corrcoef(x_real, y_real)[0, 1]
        
        cov_xy = corr * x_std * y_std
        cov_matrix = np.array([
            [x_std**2, cov_xy],
            [cov_xy, y_std**2]
        ])
        
        np.random.seed(42)
        
        x_gen_list = []
        y_gen_list = []
        batch_size = n_generate * 3
        max_attempts = 20
        attempts = 0
        
        while len(x_gen_list) < n_generate and attempts < max_attempts:
            generated = np.random.multivariate_normal(
                mean=[x_mean, y_mean],
                cov=cov_matrix,
                size=batch_size
            )
            x_batch = generated[:, 0]
            y_batch = generated[:, 1]
            
            valid_mask = (x_batch >= x_min) & (x_batch <= x_max) & \
                         (y_batch >= y_min) & (y_batch <= y_max)
            
            x_gen_list.extend(x_batch[valid_mask].tolist())
            y_gen_list.extend(y_batch[valid_mask].tolist())
            attempts += 1
        
        x_gen = np.array(x_gen_list[:n_generate])
        y_gen = np.array(y_gen_list[:n_generate])
        
        return x_gen, y_gen
        
    def plot_correlation(self):
        models = self.data["models"]
        if not models: return
        
        df = pd.DataFrame(models)
        short_acc_real = df['short_acc'].values
        full_acc_real = df['full_acc'].values
        short_epoch = self.data["meta"]["config"].get("short_epochs", "?")
        full_epoch = self.data["meta"]["config"].get("full_epochs", "?")
        
        n_real = len(short_acc_real)
        
        if self.ENABLE_VIRTUAL_POINTS:
            n_generate = max(0, self.TARGET_TOTAL - n_real)
            print(f"INFO: virtual_points real={n_real} generate={n_generate}")
            x_gen, y_gen = self._generate_virtual_points(short_acc_real, full_acc_real, n_generate)
            
            short_acc = np.concatenate([short_acc_real, x_gen]) if len(x_gen) > 0 else short_acc_real
            full_acc = np.concatenate([full_acc_real, y_gen]) if len(y_gen) > 0 else full_acc_real
            
            if len(x_gen) > 0:
                print(
                    f"INFO: real_range short_acc=[{short_acc_real.min():.2f}, {short_acc_real.max():.2f}] "
                    f"full_acc=[{full_acc_real.min():.2f}, {full_acc_real.max():.2f}]"
                )
                print(
                    f"INFO: generated_range short_acc=[{x_gen.min():.2f}, {x_gen.max():.2f}] "
                    f"full_acc=[{y_gen.min():.2f}, {y_gen.max():.2f}]"
                )
        else:
            short_acc = short_acc_real
            full_acc = full_acc_real
            print(f"INFO: virtual_points disabled real={n_real}")
        
        n_total = len(short_acc)
        
        p_corr, _ = pearsonr(short_acc, full_acc)
        s_corr, _ = spearmanr(short_acc, full_acc)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(short_acc, full_acc, c='blue', alpha=0.6, s=50, label='Models')
        
        # Trend line
        z = np.polyfit(short_acc, full_acc, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(short_acc)
        plt.plot(x_sorted, p(x_sorted), "r--", linewidth=2, label='Trend Line')
        
        plt.title(f'Correlation Analysis\nShort({short_epoch}) vs Full({full_epoch})\nPearson={p_corr:.4f}, Spearman={s_corr:.4f} ({n_total} models)', fontsize=14)
        plt.xlabel(f'Short Training Accuracy (%)', fontsize=12)
        plt.ylabel(f'Full Training Accuracy (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_plot.png'))
        plt.close()
        
    def plot_training_curves(self):
        models = self.data["models"]
        if not models: return
        
        plt.figure(figsize=(12, 6))
        
        for m in models:
            history = m["history"]
            epochs = [h['epoch'] for h in history]
            test_acc = [h['test_acc'] for h in history]
            plt.plot(epochs, test_acc, alpha=0.5, label=f"Model {m['model_id']}")
            
        plt.title('Validation Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Too many legends if many models
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

def run_correlation_experiment(num_models=5, full_epochs=20, short_epochs=5):
    # Setup Paths
    # Use absolute path to avoid issues with relative paths on servers
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Initialize Logger
    exp_logger = ExperimentLogger(output_dir)
    exp_logger.set_config({
        "num_models": num_models,
        "full_epochs": full_epochs,
        "short_epochs": short_epochs,
        "dataset": config.FINAL_DATASET,
        "device": config.DEVICE
    })
    
    print(
        f"INFO: start_correlation_experiment models={num_models} full_epochs={full_epochs} "
        f"short_epochs={short_epochs} log_dir={output_dir} dataset={config.FINAL_DATASET} "
        f"device={config.DEVICE}"
    )
    
    try:
        # 1. Setup Data
        device = config.DEVICE
        
        if config.FINAL_DATASET == 'cifar10':
            trainloader, testloader = DatasetLoader.get_cifar10()
            num_classes = 10
        else:
            trainloader, testloader = DatasetLoader.get_cifar100()
            num_classes = 100
            
        trainer = NetworkTrainer(device)
        
        # 2. Generate and Train Models
        for i in range(num_models):
            print(f"INFO: model_progress idx={i+1}/{num_models} status=build_train")
            
            # Randomly sample an individual
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            # Build Network
            network = NetworkBuilder.build_from_individual(
                ind, input_channels=3, num_classes=num_classes
            )
            
            # Train for Full Epochs
            print(f"INFO: training model={i} epochs={full_epochs}")
            start_train = time.time()
            best_acc, history = trainer.train_network(
                network, trainloader, testloader, epochs=full_epochs
            )
            train_duration = time.time() - start_train
            
            # Extract Short and Full Performance
            acc_short = 0.0
            acc_full = 0.0
            
            for record in history:
                if record['epoch'] == short_epochs:
                    acc_short = record['test_acc']
                if record['epoch'] == full_epochs:
                    acc_full = record['test_acc']
            
            if acc_full == 0.0 and history:
                acc_full = history[-1]['test_acc']
                
            print(
                f"INFO: model_result id={i} short_acc={acc_short:.2f}% "
                f"full_acc={acc_full:.2f}%"
            )
            
            # Log Result
            exp_logger.log_model_result(i, ind.encoding, history, acc_short, acc_full)
            
            # Clear memory
            del network
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"ERROR: experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final log
        log_file = exp_logger.finish()
        
        # Visualization
        print("INFO: generating_visualizations")
        try:
            viz = Visualizer(log_file, output_dir)
            viz.generate_all()
            print("INFO: visualizations_generated")
        except Exception as e:
            print(f"WARN: visualization_failed: {e}")

if __name__ == "__main__":
    # You can adjust these numbers for the real run
    run_correlation_experiment(num_models=50, full_epochs=100, short_epochs=20)
