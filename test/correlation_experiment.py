# -*- coding: utf-8 -*-
"""
Robust Experiment: Correlation between Short Training and Full Training Performance.
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

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
            
            # Periodically save to disk to prevent data loss
            if len(self.log_data["hardware_stats"]) % 10 == 0:
                self.save_log()
                
            time.sleep(interval)

    def log_model_result(self, model_id, encoding, history, short_acc, full_acc):
        with self._lock:
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
        print(f"Experiment finished. Log saved to {self.log_file}")
        return self.log_file

    def save_log(self):
        # Atomic write if possible, or just overwrite
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
        self.plot_correlation()
        self.plot_training_curves()
        self.plot_hardware_usage()
        
    def plot_correlation(self):
        models = self.data["models"]
        if not models: return
        
        df = pd.DataFrame(models)
        short_acc = df['short_acc']
        full_acc = df['full_acc']
        short_epoch = self.data["meta"]["config"].get("short_epochs", "?")
        full_epoch = self.data["meta"]["config"].get("full_epochs", "?")
        
        p_corr, _ = pearsonr(short_acc, full_acc)
        s_corr, _ = spearmanr(short_acc, full_acc)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(short_acc, full_acc, c='blue', alpha=0.7, s=100)
        
        # Trend line
        z = np.polyfit(short_acc, full_acc, 1)
        p = np.poly1d(z)
        plt.plot(short_acc, p(short_acc), "r--", linewidth=2, label='Trend Line')
        
        plt.title(f'Correlation Analysis\nShort({short_epoch}) vs Full({full_epoch})\nPearson={p_corr:.4f}, Spearman={s_corr:.4f}', fontsize=14)
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

    def plot_hardware_usage(self):
        stats = self.data["hardware_stats"]
        if not stats: return
        
        df = pd.DataFrame(stats)
        # Normalize timestamp to start from 0
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
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax2.set_ylabel('GPU Memory (MB)', color=color)  # we already handled the x-label with ax1
        ax2.plot(df['time_rel'], df['gpu_memory_allocated'], color=color, label='GPU Mem Alloc', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Hardware Resource Usage')
        plt.savefig(os.path.join(self.output_dir, 'hardware_usage.png'))
        plt.close()

def run_correlation_experiment(num_models=5, full_epochs=20, short_epochs=5):
    # Setup Paths
    output_dir = os.path.join(os.path.dirname(__file__), 'experiment_results')
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
    
    print(f"Starting Correlation Experiment")
    print(f"Models: {num_models}, Full Epochs: {full_epochs}, Short Epochs: {short_epochs}")
    print(f"Logs will be saved to {output_dir}")
    
    # Start Hardware Monitoring
    exp_logger.start_monitoring(interval=2)
    
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
            print(f"\n[{i+1}/{num_models}] Generating and training model...")
            
            # Randomly sample an individual
            ind = population_initializer.create_valid_individual()
            ind.id = i
            
            # Build Network
            network = NetworkBuilder.build_from_individual(
                ind, input_channels=3, num_classes=num_classes
            )
            
            # Train for Full Epochs
            print(f"Training model {i} for {full_epochs} epochs...")
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
                
            print(f"Model {i}: Short({short_epochs})={acc_short:.2f}%, Full({full_epochs})={acc_full:.2f}%")
            
            # Log Result
            exp_logger.log_model_result(i, ind.encoding, history, acc_short, acc_full)
            
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
    # You can adjust these numbers for the real run
    run_correlation_experiment(num_models=50, full_epochs=100, short_epochs=20)
