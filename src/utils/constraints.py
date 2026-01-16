# -*- coding: utf-8 -*-
"""
Shared helpers for enforcing parameter count bounds.
"""
from functools import lru_cache
from typing import Tuple

from configuration.config import config
from models.network import NetworkBuilder


def update_param_bounds_for_dataset(dataset: str):
    """
    Adjust global min/max param bounds based on dataset-specific overrides.
    
    Args:
        dataset (str): The dataset name (e.g., 'cifar10', 'imagenet').
    """
    bounds = config.DATASET_PARAM_BOUNDS.get(dataset, None)
    if bounds:
        min_val = bounds.get("min", config.MIN_PARAM_COUNT)
        max_val = bounds.get("max", config.MAX_PARAM_COUNT)
        config.MIN_PARAM_COUNT = min_val
        config.MAX_PARAM_COUNT = max_val


def check_param_bounds(param_count: int) -> Tuple[bool, str]:
    """
    Validate parameter count against configured min/max bounds.

    Args:
        param_count (int): The number of parameters.
    
    Returns:
        Tuple[bool, str]: (isValid, failure_reason)
    """
    if param_count < config.MIN_PARAM_COUNT:
        return False, f"params {param_count} < minimum {config.MIN_PARAM_COUNT}"
    if param_count > config.MAX_PARAM_COUNT:
        return False, f"params {param_count} > maximum {config.MAX_PARAM_COUNT}"
    return True, ""


def _encoding_to_tuple(encoding: list) -> Tuple[int, ...]:
    return tuple(int(x) for x in encoding)


@lru_cache(maxsize=4096)
def _cached_param_count(enc_tuple: Tuple[int, ...], input_channels: int, num_classes: int) -> int:
    encoding_list = list(enc_tuple)
    network = NetworkBuilder.build_from_encoding(encoding_list, input_channels, num_classes)
    return network.get_param_count()


def evaluate_encoding_params(
    encoding: list,
    input_channels: int = None,
    num_classes: int = None,
) -> Tuple[bool, str, int]:
    """
    Compute parameter count for an encoding and return validity, reason, and count.

    Calculates the parameter count by temporarily building the model.
    Falls back to dataset-aware defaults when channels/classes are not provided.

    Args:
        encoding (list): Architecture encoding.
        input_channels (int, optional): Input channel dimension.
        num_classes (int, optional): Output class count.

    Returns:
        Tuple[bool, str, int]: (isValid, failure_reason, param_count)
    """
    if input_channels is None:
        input_channels = 3
    if num_classes is None:
        num_classes = config.NTK_NUM_CLASSES

    enc_tuple = _encoding_to_tuple(encoding)
    try:
        param_count = _cached_param_count(enc_tuple, input_channels, num_classes)
    except Exception as exc:
        # Return a failure tuple instead of crashing the search loop.
        return False, f"failed to build network for param count: {exc}", 0

    ok, reason = check_param_bounds(param_count)
    return ok, reason, param_count
