# -*- coding: utf-8 -*-
"""
日志模块
负责日志记录和失败个体记录
"""
import os
import logging
import sys
import time
from configuration.config import config


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

    def log_generation(self, gen, best_fitness, avg_fitness, pop_size):
        """记录每代统计信息"""
        self.info(f"Gen {gen}: Best Fitness={best_fitness:.6f}, Avg Fitness={avg_fitness:.6f}, Pop Size={pop_size}")
        
    def log_evaluation(self, ind_id, eval_type, score, param_count=None):
        """记录评估结果"""
        msg = f"Eval {ind_id} ({eval_type}): {score}"
        if param_count is not None:
            msg += f", Params: {param_count}"
        self.info(msg)

    def log_mutation(self, mut_type, old_id, new_id):
        """记录变异操作"""
        self.debug(f"Mutation {mut_type}: {old_id} -> {new_id}")


class FailedLogger:
    """失败个体记录器"""
    def save_failed_individual(self, ind, reason, gen):
        pass


logger = Logger()
failed_logger = FailedLogger()
