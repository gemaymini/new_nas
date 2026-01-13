# -*- coding: utf-8 -*-
"""
Inspect a saved model encoding and architecture.
"""

import torch
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoding import Individual, Encoder
from models.network import NetworkBuilder

def inspect_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"ERROR: model file not found: {model_path}")
        return

    print(f"INFO: loading_model={model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR: failed to load checkpoint: {e}")
        return

    encoding = None
    state_dict = None
    metadata = {}

    if isinstance(checkpoint, dict):
        if 'encoding' in checkpoint:
            encoding = checkpoint['encoding']
            state_dict = checkpoint.get('state_dict')
            for k, v in checkpoint.items():
                if k not in ['state_dict', 'encoding', 'history']:
                    metadata[k] = v
        elif 'state_dict' in checkpoint:
            print("WARN: checkpoint only contains state_dict (legacy format)")
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print("ERROR: unknown checkpoint format")
        return

    if encoding:
        print(f"INFO: metadata items={len(metadata)}")
        for k, v in metadata.items():
            print(f"INFO: meta {k}={v}")

        print(f"INFO: encoding={encoding}")
        print("INFO: architecture_details")
        Encoder.print_architecture(encoding)
        
        try:
            print("INFO: network_structure")
            ind = Individual(encoding)
            
            num_classes = 10
            if state_dict:
                if 'fc.weight' in state_dict:
                    num_classes = state_dict['fc.weight'].shape[0]
                elif 'classifier.weight' in state_dict:
                    num_classes = state_dict['classifier.weight'].shape[0]
            
            network = NetworkBuilder.build_from_individual(
                ind, input_channels=3, num_classes=num_classes
            )
            for line in str(network).splitlines():
                print(f"INFO: {line}")
            
            param_count = network.get_param_count()
            print(f"INFO: total_params={param_count:,}")
            
        except Exception as e:
            print(f"WARN: network build failed: {e}")

    else:
        print("ERROR: no encoding found in checkpoint; cannot reconstruct architecture")
        print("INFO: checkpoint appears to be legacy state_dict")
        
        if state_dict:
            print("INFO: state_dict_keys first=20")
            for k in list(state_dict.keys())[:20]:
                print(f"INFO: key={k}")
            if len(state_dict) > 20:
                print(f"INFO: key=... remaining={len(state_dict) - 20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect NAS Model Architecture')
    parser.add_argument('model_path', type=str, help='Path to model .pth file')
    
    args = parser.parse_args()
    inspect_model(args.model_path)
