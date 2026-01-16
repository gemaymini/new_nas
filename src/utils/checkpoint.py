# -*- coding: utf-8 -*-
"""
Utilities for loading checkpoints and extracting model information.
"""
import os
import torch
import ast
from typing import Dict, Any, Optional

def load_checkpoint(model_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a model checkpoint from a file.
    
    Args:
        model_path: Path to the .pth file.
        device: Device to map location to ('cpu' or 'cuda').
        
    Returns:
        The loaded checkpoint dictionary.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If loading fails or format is invalid.
        
    Security Warning:
        This function uses `torch.load` which implicitly uses `pickle`. 
        Only load checkpoints from trusted sources. serialized data can execute arbitrary code.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")
        
    return checkpoint

def extract_encoding_from_checkpoint(checkpoint: Any, encoding_str: Optional[str] = None) -> list:
    """
    Extract the architecture encoding from a checkpoint or provided string.
    
    Args:
        checkpoint: Loaded checkpoint object (dict or other).
        encoding_str: Optional string representation of encoding (fallback).
        
    Returns:
        List[int]: The architecture encoding.
        
    Raises:
        ValueError: If encoding cannot be found or parsed.
    """
    encoding = None
    
    if isinstance(checkpoint, dict):
        if 'encoding' in checkpoint:
            encoding = checkpoint['encoding']
        elif 'model_encoding' in checkpoint: # Handle potential variations
            encoding = checkpoint['model_encoding']
            
    if encoding is None and encoding_str:
        try:
            encoding = ast.literal_eval(encoding_str)
        except Exception as e:
            raise ValueError(f"Failed to parse provided encoding string: {e}")
            
    if encoding is None:
        raise ValueError("Could not find 'encoding' in checkpoint and no valid encoding string provided.")
        
    if not isinstance(encoding, list):
         raise ValueError(f"Extracted encoding is not a list: {type(encoding)}")
         
    return encoding

def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    """
    Extract the state_dict from a checkpoint.
    """
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        # Check if the checkpoint itself is a state dict (contains tensor weights)
        if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
             return checkpoint
             
    raise ValueError("Could not extract state_dict from checkpoint.")
