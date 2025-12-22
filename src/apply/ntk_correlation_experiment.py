# -*- coding: utf-8 -*-
"""
Experiment: Correlation between NTK Condition Number and Short Training Performance.
Includes comprehensive logging, hardware monitoring, and visualization.
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
import threading
import psutil
import torch
import hashlib

# Add src to path (apply is now under src/apply/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config
from core.search_space import population_initializer
from models.network import NetworkBuilder
from engine.trainer import NetworkTrainer
from engine.evaluator import NTKEvaluator
from data.dataset import DatasetLoader
from utils.logger import logger

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'ntk_experiment_log_{timestamp}.json')
        
        self.log_data = {
            "meta": {
                "start_time": str(datetime.datetime.now()),
                "end_time": None,
                "config": {},
                "status": "running"
            },
            "models": [],
            "hardware_stats": []
        }
        
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
    def start_monitoring(self, interval=5):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self, interval):
        while self.monitoring:
            stats = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "gpu_memory_allocated": 0,
                "gpu_memory_reserved": 0
            }
            
            if torch.cuda.is_available():
                stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**2) # MB
                stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**2) # MB
                
            with self._lock:
                self.log_data["hardware_stats"].append(stats)
            
            # Periodically save to disk
            if len(self.log_data["hardware_stats"]) % 10 == 0:
                self.save_log()
                
            time.sleep(interval)

    def log_model_result(self, model_id, encoding, history, ntk_cond, short_acc):
        with self._lock:
            self.log_data["models"].append({
                "model_id": model_id,
                "encoding": str(encoding),
                "ntk_cond": ntk_cond,
                "short_acc": short_acc,
                "history": history
            })
        self.save_log()

    def set_config(self, config_dict):
        self.log_data["meta"]["config"] = config_dict
        
    def finish(self):
        self.log_data["meta"]["end_time"] = str(datetime.datetime.now())
        self.log_data["meta"]["status"] = "completed"
        
        content = json.dumps(self.log_data, sort_keys=True)
        checksum = hashlib.md5(content.encode('utf-8')).hexdigest()
        self.log_data["meta"]["checksum"] = checksum
        
        self.save_log()
        print(f"Experiment finished. Log saved to {self.log_file}")
        return self.log_file

    def save_log(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")

class Visualizer:
    def __init__(self, log_file, output_dir):
        self.log_file = log_file
        self.output_dir = output_dir
        with open(log_file, 'r') as f:
            self.data = json.load(f)
            
    def generate_all(self):
        self.plot_ntk_correlation()
        self.plot_training_curves()
        self.plot_hardware_usage()
        
    def plot_ntk_correlation(self):
        models = self.data["models"]
        if not models: return
        
        df = pd.DataFrame(models)
        ntk_cond = df['ntk_cond']
        short_acc = df['short_acc']
        short_epoch = self.data["meta"]["config"].get("short_epochs", "?")
        
        # Calculate correlation (Log Cond vs Acc usually linear)
        # Using Log(Cond) because Condition Number spans orders of magnitude
        log_cond = np.log10(ntk_cond.replace(0, 1e-6)) # Avoid log(0)
        
        p_corr, _ = pearsonr(log_cond, short_acc)
        s_corr, _ = spearmanr(ntk_cond, short_acc) # Spearman is rank-based, so log doesn't matter
        
        plt.figure(figsize=(10, 8))
        plt.scatter(log_cond, short_acc, c='green', alpha=0.7, s=100)
        
        # Trend line
        z = np.polyfit(log_cond, short_acc, 1)
        p = np.poly1d(z)
        plt.plot(log_cond, p(log_cond), "r--", linewidth=2, label='Trend Line')
        
        plt.title(f'Correlation Analysis\nLog10(NTK Cond) vs Short Acc({short_epoch})\nPearson(Log)={p_corr:.4f}, Spearman={s_corr:.4f}', fontsize=14)
        plt.xlabel(f'Log10(NTK Condition Number) (Lower is better)', fontsize=12)
        plt.ylabel(f'Short Training Accuracy (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ntk_correlation_plot.png'))
        plt.close()
        
    def plot_training_curves(self):
        models = self.data["models"]
        if not models: return
        
        plt.figure(figsize=(12, 6))
        
        for m in models:
            history = m["history"]
            epochs = [h['epoch'] for h in history]
            test_acc = [h['test_acc'] for h in history]
            plt.plot(epochs, test_acc, alpha=0.5, label=f"Model {m['model_id']} (Cond={m['ntk_cond']:.1e})")
            
        plt.title('Validation Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

    def plot_hardware_usage(self):
        stats = self.data["hardware_stats"]
        if not stats: return
        
        df = pd.DataFrame(stats)
        start_time = df['timestamp'].iloc[0]
        df['time_rel'] = df['timestamp'] - start_time
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU/Memory Usage (%)', color=color)
        ax1.plot(df['time_rel'], df['cpu_percent'], color=color, label='CPU %', alpha=0.6)
        ax1.plot(df['time_rel'], df['memory_percent'], color='tab:orange', label='RAM %', alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('GPU Memory (MB)', color=color)
        ax2.plot(df['time_rel'], df['gpu_memory_allocated'], color=color, label='GPU Mem Alloc', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('Hardware Resource Usage')
        plt.savefig(os.path.join(self.output_dir, 'hardware_usage.png'))
        plt.close()

def run_ntk_experiment(num_models=5, short_epochs=5):
    # Setup Paths
    # Use absolute path to avoid issues with relative paths on servers
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ntk_experiment_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Initialize Logger
    exp_logger = ExperimentLogger(output_dir)
    exp_logger.set_config({
        "num_models": num_models,
        "short_epochs": short_epochs,
        "dataset": config.FINAL_DATASET,
        "device": config.DEVICE
    })
    
    print(f"Starting NTK Correlation Experiment")
    print(f"Models: {num_models}, Short Epochs: {short_epochs}")
    print(f"Logs will be saved to {output_dir}")
    
    # Start Hardware Monitoring
    exp_logger.start_monitoring(interval=2)
    
    try:
        # 1. Setup Data & Components
        device = config.DEVICE
        if config.FINAL_DATASET == 'cifar10':
            trainloader, testloader = DatasetLoader.get_cifar10()
            num_classes = 10
        else:
            trainloader, testloader = DatasetLoader.get_cifar100()
            num_classes = 100
            
        trainer = NetworkTrainer(device)
        ntk_evaluator = NTKEvaluator()
        
        # 2. Generate and Evaluate Models
        for i in range(num_models):
            print(f"\n[{i+1}/{num_models}] Generating and evaluating model...")
            
            # Randomly sample
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            # A. Calculate NTK
            print(f"Calculating NTK for model {i}...")
            ntk_evaluator.evaluate_individual(ind)
            ntk_cond = ind.fitness  # fitness 直接等于 NTK 条件数（越小越好）
                
            print(f"Model {i}: NTK Cond={ntk_cond:.2f}")
            
            # B. Short Training
            network = NetworkBuilder.build_from_individual(
                ind, input_channels=3, num_classes=num_classes
            )
            
            print(f"Training model {i} for {short_epochs} epochs...")
            start_train = time.time()
            best_acc, history = trainer.train_network(
                network, trainloader, testloader, epochs=short_epochs
            )
            
            # Extract Acc
            short_acc = 0.0
            if history:
                short_acc = history[-1]['test_acc'] # Last epoch accuracy
                
            print(f"Model {i}: Short Acc={short_acc:.2f}%")
            
            # Log Result
            exp_logger.log_model_result(i, ind.encoding, history, ntk_cond, short_acc)
            
            # Clear memory
            del network
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop monitoring and save final log
        exp_logger.stop_monitoring()
        log_file = exp_logger.finish()
        
        # Visualization
        print("Generating visualizations...")
        try:
            viz = Visualizer(log_file, output_dir)
            viz.generate_all()
            print("Visualizations generated.")
        except Exception as e:
            print(f"Visualization failed: {e}")

if __name__ == "__main__":
    run_ntk_experiment(num_models=10, short_epochs=15)
