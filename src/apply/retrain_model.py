# -*- coding: utf-8 -*-
"""
Retrain a saved architecture multiple times and summarize results.
"""

import torch
import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoding import Individual, Encoder
from models.network import NetworkBuilder
from data.dataset import DatasetLoader
from engine.trainer import NetworkTrainer
from configuration.config import config
from utils.logger import logger


def load_model_encoding(model_path: str) -> list:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"INFO: loading_model={model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    encoding = None
    
    if isinstance(checkpoint, dict):
        if 'encoding' in checkpoint:
            encoding = checkpoint['encoding']
            print("INFO: encoding_found")
            
            for k, v in checkpoint.items():
                if k not in ['state_dict', 'encoding', 'history']:
                    print(f"INFO: meta {k}={v}")
        else:
            raise ValueError("Checkpoint does not contain 'encoding' key. Cannot extract model structure.")
    else:
        raise ValueError("Unknown checkpoint format. Expected a dict with 'encoding' key.")
    
    return encoding


def build_fresh_network(encoding: list, input_channels: int = 3, num_classes: int = 10):
    network = NetworkBuilder.build_from_encoding(encoding, input_channels, num_classes)
    return network


def train_once(encoding: list, trainloader, testloader, epochs: int, 
               run_id: int, device: str = None) -> dict:
    print(f"INFO: training_run_start id={run_id}")
    
    network = build_fresh_network(encoding)
    param_count = network.get_param_count()
    print(f"INFO: network_params={param_count:,}")
    
    trainer = NetworkTrainer(device=device)
    
    best_acc, history = trainer.train_network(
        model=network,
        trainloader=trainloader,
        testloader=testloader,
        epochs=epochs
    )
    
    final_train_acc = history[-1]['train_acc']
    final_test_acc = history[-1]['test_acc']
    
    print(
        f"INFO: training_run_complete id={run_id} best_acc={best_acc:.2f}% "
        f"final_train_acc={final_train_acc:.2f}% final_test_acc={final_test_acc:.2f}%"
    )
    
    return {
        'run_id': run_id,
        'best_acc': best_acc,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'param_count': param_count,
        'history': history
    }


def retrain_model(model_path: str, epochs: int = 300, num_runs: int = 10,
                  dataset: str = 'cifar10', device: str = None,
                  save_results: bool = True) -> dict:
    encoding = load_model_encoding(model_path)
    
    print(f"INFO: encoding={encoding}")
    print("INFO: architecture_details")
    Encoder.print_architecture(encoding)
    
    print("INFO: loading_dataset")
    if dataset == 'cifar10':
        trainloader, testloader = DatasetLoader.get_cifar10()
    elif dataset == 'cifar100':
        trainloader, testloader = DatasetLoader.get_cifar100()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"INFO: dataset={dataset} train_batches={len(trainloader)} test_batches={len(testloader)}")
    
    if device is None:
        device = config.DEVICE
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("WARN: cuda not available; using cpu")
    print(f"INFO: device={device}")
    
    all_results = []
    best_accs = []
    final_test_accs = []
    
    start_time = datetime.now()
    
    try:
        for run in range(1, num_runs + 1):
            result = train_once(
                encoding=encoding,
                trainloader=trainloader,
                testloader=testloader,
                epochs=epochs,
                run_id=run,
                device=device
            )
            all_results.append(result)
            best_accs.append(result['best_acc'])
            final_test_accs.append(result['final_test_acc'])
    except KeyboardInterrupt:
        print("WARN: training interrupted by user; saving partial results")
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        runs_completed = len(all_results)
        best_acc_mean = np.mean(best_accs) if best_accs else float('nan')
        best_acc_std = np.std(best_accs) if best_accs else float('nan')
        final_acc_mean = np.mean(final_test_accs) if final_test_accs else float('nan')
        final_acc_std = np.std(final_test_accs) if final_test_accs else float('nan')
        
        summary = {
            'status': 'interrupted',
            'model_path': model_path,
            'encoding': encoding,
            'dataset': dataset,
            'epochs': epochs,
            'num_runs': num_runs,
            'runs_completed': runs_completed,
            'param_count': all_results[0]['param_count'] if all_results else None,
            'best_acc_mean': best_acc_mean,
            'best_acc_std': best_acc_std,
            'final_acc_mean': final_acc_mean,
            'final_acc_std': final_acc_std,
            'best_accs': best_accs,
            'final_test_accs': final_test_accs,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'all_results': [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
        }
        
        result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retrain_results')
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"retrain_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_interrupted.json"
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"INFO: partial_results_saved={result_path}")
        
        return summary
    except Exception as e:
        print(f"ERROR: training failed: {e}; saving partial results")
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        runs_completed = len(all_results)
        best_acc_mean = np.mean(best_accs) if best_accs else float('nan')
        best_acc_std = np.std(best_accs) if best_accs else float('nan')
        final_acc_mean = np.mean(final_test_accs) if final_test_accs else float('nan')
        final_acc_std = np.std(final_test_accs) if final_test_accs else float('nan')
        
        summary = {
            'status': 'error',
            'error_message': str(e),
            'model_path': model_path,
            'encoding': encoding,
            'dataset': dataset,
            'epochs': epochs,
            'num_runs': num_runs,
            'runs_completed': runs_completed,
            'param_count': all_results[0]['param_count'] if all_results else None,
            'best_acc_mean': best_acc_mean,
            'best_acc_std': best_acc_std,
            'final_acc_mean': final_acc_mean,
            'final_acc_std': final_acc_std,
            'best_accs': best_accs,
            'final_test_accs': final_test_accs,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'all_results': [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
        }
        
        result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retrain_results')
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"retrain_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_error.json"
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"INFO: partial_results_saved={result_path}")
        
        return summary
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"INFO: training_summary runs={num_runs} completed={len(all_results)}")
    
    best_acc_mean = np.mean(best_accs) if best_accs else float('nan')
    best_acc_std = np.std(best_accs) if best_accs else float('nan')
    final_acc_mean = np.mean(final_test_accs) if final_test_accs else float('nan')
    final_acc_std = np.std(final_test_accs) if final_test_accs else float('nan')
    
    print(
        f"INFO: summary_stats best_acc_mean={best_acc_mean:.2f}% "
        f"best_acc_std={best_acc_std:.2f}% final_acc_mean={final_acc_mean:.2f}% "
        f"final_acc_std={final_acc_std:.2f}%"
    )
    print(f"INFO: best_accs={[f'{acc:.2f}%' for acc in best_accs]}")
    print(f"INFO: final_accs={[f'{acc:.2f}%' for acc in final_test_accs]}")
    if best_accs:
        print(f"INFO: best_acc_min={min(best_accs):.2f}% best_acc_max={max(best_accs):.2f}%")
    else:
        print("WARN: no runs completed; min/max unavailable")
    print(f"INFO: total_time_hours={total_time/3600:.2f}")
    print(f"INFO: avg_time_minutes={total_time/num_runs/60:.2f}")
    
    summary = {
        'status': 'completed',
        'model_path': model_path,
        'encoding': encoding,
        'dataset': dataset,
        'epochs': epochs,
        'num_runs': num_runs,
        'runs_completed': len(all_results),
        'param_count': all_results[0]['param_count'] if all_results else None,
        'best_acc_mean': best_acc_mean,
        'best_acc_std': best_acc_std,
        'final_acc_mean': final_acc_mean,
        'final_acc_std': final_acc_std,
        'best_accs': best_accs,
        'final_test_accs': final_test_accs,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'all_results': [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
    }
    
    if save_results:
        result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retrain_results')
        os.makedirs(result_dir, exist_ok=True)
        
        result_filename = f"retrain_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(result_dir, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"INFO: results_saved={result_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Retrain a model structure multiple times and compute average performance')
    parser.add_argument('model_path', type=str, help='Path to the .pth model file')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs per run (default: 300)')
    parser.add_argument('--runs', type=int, default=3, help='Number of training runs (default: 10)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], 
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use for training (default: auto)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    retrain_model(
        model_path=args.model_path,
        epochs=args.epochs,
        num_runs=args.runs,
        dataset=args.dataset,
        device=args.device,
        save_results=not args.no_save
    )


if __name__ == '__main__':
    main()
