# -*- coding: utf-8 -*-
"""
Tests for utils.logger.
"""
from pathlib import Path

from utils.logger import OperationLogger, Logger


def test_operation_logger(tmp_path):
    op_logger = OperationLogger(str(tmp_path))
    record = {"step": 1, "child_id": 2, "fitness": 3.0}
    op_logger.log(record)
    assert (tmp_path / "op_history.jsonl").exists()


def test_logger_methods(tmp_path):
    logger = Logger()
    logger.log_architecture(1, [1, 2, 3], fitness=0.1, param_count=10)
    logger.log_generation(1, 0.1, 0.2, 5)
    logger.log_unit_stats(1, {2: 3})
    logger.log_evaluation(1, "NTK", 0.1, param_count=10)
    logger.log_mutation("mutate", 1, 2, details={"x": 1}, step=1)
