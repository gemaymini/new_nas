# -*- coding: utf-8 -*-
"""
日志模块
负责日志记录、TensorBoard可视化和失败个体记录
"""
import os
import logging
import sys
import time
from .config import config

class Logger:
    """
    统一日志记录器
    """
    def __init__(self):
        self.logger = logging.getLogger('NAS')
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # 确保日志目录存在
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR)
            
        # 文件处理器
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(config.LOG_DIR, f'nas_{timestamp}.log'),
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(message)s'
        ))
        self.logger.addHandler(console_handler)
        
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
        
    def log_phase_change(self, gen, phase):
        """记录阶段切换"""
        self.info(f"Generation {gen}: Switching to Phase {phase}")

    def log_evaluation(self, ind_id, eval_type, score):
        """记录评估结果"""
        self.debug(f"Eval {ind_id} ({eval_type}): {score}")

    def log_mutation(self, mut_type, old_id, new_id):
        """记录变异操作"""
        self.debug(f"Mutation {mut_type}: {old_id} -> {new_id}")


# TensorBoard Logger Stub (Simplify for now, or implement if needed)
class TBLogger:
    def __init__(self):
        self.writer = None
        if config.USE_TENSORBOARD:
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

    def log_pareto_front(self, gen, pareto_front):
        # Implementation skipped for brevity
        pass

    def close(self):
        if self.writer:
            self.writer.close()

# Failed Individuals Logger Stub
class FailedLogger:
    def save_failed_individual(self, ind, reason, gen):
        # Implementation skipped for brevity
        pass

logger = Logger()
tb_logger = TBLogger()
failed_logger = FailedLogger()
