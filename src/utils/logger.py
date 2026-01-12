# -*- coding: utf-8 -*-
"""
日志模块
负责日志记录、TensorBoard可视化和失败个体记录
"""
import os
import logging
import sys
import time
import json
from configuration.config import config


class OperationLogger:
    """
    负责将变异/交叉操作按 JSONL 记录，便于离线分析
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.filepath = os.path.join(self.log_dir, 'op_history.jsonl')

    def log(self, record: dict):
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Operation log failed: {e}")

class Logger:
    """
    统一日志记录器
    """
    def __init__(self):
        self.logger = logging.getLogger('NAS')
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        self.file_handler = None
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(message)s'
        ))
        self.logger.addHandler(console_handler)
        
    def setup_file_logging(self):
        """
        初始化文件日志记录
        """
        if self.file_handler is not None:
            return
            
        # 确保日志目录存在
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR)
            
        # 文件处理器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_handler = logging.FileHandler(
            os.path.join(config.LOG_DIR, f'nas_{timestamp}.log'),
            encoding='utf-8'
        )
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(self.file_handler)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def log_architecture(self, ind_id, encoding, fitness=None, param_count=None, prefix=""):
        """记录架构信息"""
        msg = f"{prefix}ID: {ind_id}, Fitness: {fitness}, Params: {param_count}\n"
        msg += f"Encoding: {encoding}"
        self.info(msg)

    def log_generation(self, gen, best_fitness, avg_fitness, pop_size):
        """记录每代统计信息"""
        self.info(f"Gen {gen}: Best Fitness={best_fitness:.6f}, Avg Fitness={avg_fitness:.6f}, Pop Size={pop_size}")

    def log_unit_stats(self, gen, unit_counts):
        """记录每代Unit数量统计"""
        msg = f"Gen {gen} Unit Stats: "
        stats_str = ", ".join([f"{k} units: {v}" for k, v in sorted(unit_counts.items())])
        self.info(msg + stats_str)
        
    def log_evaluation(self, ind_id, eval_type, score, param_count=None):
        """记录评估结果"""
        msg = f"Eval {ind_id} ({eval_type}): {score}"
        if param_count is not None:
            msg += f", Params: {param_count}"
        self.info(msg)

    def log_mutation(self, mut_type, old_id, new_id, details=None, step=None):
        """记录变异操作"""
        prefix = f"[Step {step}] " if step is not None else ""
        msg = f"{prefix}Mutation {mut_type}: {old_id} -> {new_id}"
        if details:
            msg += f" | details: {details}"
        self.debug(msg)

    def log_operation(self, record: dict):
        """
        记录一条完整的交叉/变异操作日志并写入 JSONL
        """
        step = record.get("step")
        child_id = record.get("child_id")
        fitness = record.get("fitness")
        self.info(f"Step {step}: child {child_id} fitness={fitness}")
        try:
            op_logger.log(record)
        except Exception as e:
            self.warning(f"Failed to persist op log: {e}")


# TensorBoard Logger Stub (Simplify for now, or implement if needed)
class TBLogger:
    def __init__(self):
        self.writer = None
        
    def setup(self):
        if config.USE_TENSORBOARD and self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if not os.path.exists(config.TENSORBOARD_DIR):
                    os.makedirs(config.TENSORBOARD_DIR)
                self.writer = SummaryWriter(config.TENSORBOARD_DIR)
            except ImportError:
                print("TensorBoard not installed, skipping.")

    def log_generation_stats(self, gen, stats):
        if self.writer:
            for key, value in stats.items():
                self.writer.add_scalar(f"Stats/{key}", value, gen)

    def close(self):
        if self.writer:
            self.writer.close()

# Failed Individuals Logger Stub
class FailedLogger:
    def save_failed_individual(self, ind, reason, gen):
        # Implementation skipped for brevity
        pass

op_logger = OperationLogger(config.LOG_DIR)
logger = Logger()
tb_logger = TBLogger()
failed_logger = FailedLogger()
