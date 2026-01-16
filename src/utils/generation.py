# -*- coding: utf-8 -*-
"""
Helpers for generating valid offspring under encoding and param-count constraints.
"""
import random
from typing import Callable, Optional

from configuration.config import config
from core.encoding import Encoder, Individual
from utils.constraints import evaluate_encoding_params
from utils.logger import logger


def generate_valid_child(
    parent1: Individual,
    parent2: Individual,
    crossover_fn: Callable,
    mutation_fn: Callable,
    repair_fn: Optional[Callable],
    resample_fn: Callable[[], Individual],
    crossover_prob: float = None,
    mutation_prob: float = None,
    max_attempts: int = 50,
) -> Individual:
    """
    Generate an offspring that passes encoding validation and param bounds.

    Args:
        parent1 (Individual): First parent.
        parent2 (Individual): Second parent.
        crossover_fn (Callable): Function to perform crossover.
        mutation_fn (Callable): Function to perform mutation.
        repair_fn (Optional[Callable]): Function to repair invalid encodings.
        resample_fn (Callable): Function to generate a fresh individual if needed.
        crossover_prob (float, optional): Probability of crossover.
        mutation_prob (float, optional): Probability of mutation.
        max_attempts (int): Maximum attempts to generate a valid child before falling back to resample.

    Returns:
        Individual: A valid offspring.
    """
    crossover_prob = crossover_prob if crossover_prob is not None else config.PROB_CROSSOVER
    mutation_prob = mutation_prob if mutation_prob is not None else config.PROB_MUTATION

    attempts = 0
    last_reason = ""

    while attempts < max_attempts:
        attempts += 1
        ops = []

        do_crossover = random.random() < crossover_prob
        do_mutation = random.random() < mutation_prob or not do_crossover

        if do_crossover:
            c1, c2, cross_detail = crossover_fn(parent1, parent2)
            chosen_child, chosen_label = random.choice([(c1, "child1"), (c2, "child2")])
            if cross_detail is None:
                cross_detail = {}
            cross_detail["parent_ids"] = [parent1.id, parent2.id]
            cross_detail["chosen_child"] = chosen_label
            cross_detail["op"] = "crossover"
            ops.append(cross_detail)
            child = chosen_child
        else:
            chosen_parent = random.choice([parent1, parent2])
            child = chosen_parent.copy()
            ops.append(
                {
                    "op": "copy_parent",
                    "applied": True,
                    "source_parent_id": chosen_parent.id,
                }
            )

        if do_mutation:
            child = mutation_fn(child)
            if hasattr(child, "op_history"):
                ops.extend(child.op_history)

        child.op_history = ops

        if not Encoder.validate_encoding(child.encoding):
            if repair_fn is not None:
                child = repair_fn(child)
                repair_ops = getattr(child, "op_history", [])
                child.op_history = ops + repair_ops
            else:
                last_reason = "encoding invalid after generate/mutate"
                continue

        ok, reason, param_count = evaluate_encoding_params(child.encoding)
        if ok:
            child.param_count = param_count
            return child

        last_reason = reason
        logger.warning(f"Child param bounds failed ({reason}); regenerating")

    logger.warning("Too many invalid children; sampling fresh individual")
    child = resample_fn()
    child.op_history = child.op_history + [
        {"op": "resample_for_param_bounds", "reason": last_reason}
    ] if hasattr(child, "op_history") else [
        {"op": "resample_for_param_bounds", "reason": last_reason}
    ]
    return child
