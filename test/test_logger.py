# -*- coding: utf-8 -*-
"""
单元测试 - 日志模块 (logger.py)
测试 Logger, TBLogger, FailedLogger 类
"""
import sys
import os
import unittest
import tempfile
import shutil

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.logger import Logger, TBLogger, FailedLogger, logger, tb_logger, failed_logger


class TestLogger(unittest.TestCase):
    """测试 Logger 类"""
    
    def test_init(self):
        """测试初始化"""
        log = Logger()
        self.assertIsNotNone(log.logger)
    
    def test_info(self):
        """测试 info 级别日志"""
        log = Logger()
        # 不应该抛出异常
        try:
            log.info("Test info message")
        except Exception as e:
            self.fail(f"info raised exception: {e}")
    
    def test_debug(self):
        """测试 debug 级别日志"""
        log = Logger()
        try:
            log.debug("Test debug message")
        except Exception as e:
            self.fail(f"debug raised exception: {e}")
    
    def test_warning(self):
        """测试 warning 级别日志"""
        log = Logger()
        try:
            log.warning("Test warning message")
        except Exception as e:
            self.fail(f"warning raised exception: {e}")
    
    def test_error(self):
        """测试 error 级别日志"""
        log = Logger()
        try:
            log.error("Test error message")
        except Exception as e:
            self.fail(f"error raised exception: {e}")
    
    def test_log_architecture(self):
        """测试记录架构信息"""
        log = Logger()
        try:
            log.log_architecture(
                ind_id=1, 
                encoding=[1, 2, 3], 
                fitness=0.5, 
                param_count=1000,
                prefix="Test: "
            )
        except Exception as e:
            self.fail(f"log_architecture raised exception: {e}")
    
    def test_log_generation(self):
        """测试记录每代信息"""
        log = Logger()
        try:
            log.log_generation(gen=1, best_fitness=0.9, avg_fitness=0.5, pop_size=100)
        except Exception as e:
            self.fail(f"log_generation raised exception: {e}")
    
    def test_log_unit_stats(self):
        """测试记录 unit 统计"""
        log = Logger()
        try:
            log.log_unit_stats(gen=1, unit_counts={3: 50, 4: 30, 5: 20})
        except Exception as e:
            self.fail(f"log_unit_stats raised exception: {e}")
    
    def test_log_evaluation(self):
        """测试记录评估结果"""
        log = Logger()
        try:
            log.log_evaluation(ind_id=1, eval_type="NTK", score=-0.5)
        except Exception as e:
            self.fail(f"log_evaluation raised exception: {e}")
    
    def test_log_mutation(self):
        """测试记录变异操作"""
        log = Logger()
        try:
            log.log_mutation(mut_type="swap_blocks", old_id=1, new_id=2)
        except Exception as e:
            self.fail(f"log_mutation raised exception: {e}")
    
    def test_setup_file_logging(self):
        """测试设置文件日志"""
        # 使用临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 临时修改配置
            from configuration.config import config
            original_log_dir = config.LOG_DIR
            config.LOG_DIR = temp_dir
            
            log = Logger()
            log.setup_file_logging()
            
            # 检查文件是否创建
            log_files = [f for f in os.listdir(temp_dir) if f.endswith('.log')]
            self.assertGreater(len(log_files), 0)
            
            # 恢复配置
            config.LOG_DIR = original_log_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestTBLogger(unittest.TestCase):
    """测试 TBLogger 类"""
    
    def test_init(self):
        """测试初始化"""
        tb = TBLogger()
        self.assertIsNone(tb.writer)
    
    def test_log_generation_stats_without_setup(self):
        """测试在未设置情况下记录统计"""
        tb = TBLogger()
        # 不应该抛出异常
        try:
            tb.log_generation_stats(gen=1, stats={'best_fitness': 0.9})
        except Exception as e:
            self.fail(f"log_generation_stats raised exception: {e}")
    
    def test_close_without_setup(self):
        """测试在未设置情况下关闭"""
        tb = TBLogger()
        try:
            tb.close()
        except Exception as e:
            self.fail(f"close raised exception: {e}")
    
    def test_setup(self):
        """测试设置 TensorBoard"""
        # 使用临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            from configuration.config import config
            original_tb_dir = config.TENSORBOARD_DIR
            original_use_tb = config.USE_TENSORBOARD
            
            config.TENSORBOARD_DIR = temp_dir
            config.USE_TENSORBOARD = True
            
            tb = TBLogger()
            tb.setup()
            
            # 如果 TensorBoard 可用，writer 应该被创建
            # 如果不可用，writer 仍然是 None
            
            # 恢复配置
            config.TENSORBOARD_DIR = original_tb_dir
            config.USE_TENSORBOARD = original_use_tb
            
            tb.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFailedLogger(unittest.TestCase):
    """测试 FailedLogger 类"""
    
    def test_save_failed_individual(self):
        """测试保存失败个体"""
        fl = FailedLogger()
        # 不应该抛出异常
        try:
            fl.save_failed_individual(ind=None, reason="test", gen=1)
        except Exception as e:
            self.fail(f"save_failed_individual raised exception: {e}")


class TestGlobalLoggerInstances(unittest.TestCase):
    """测试全局日志实例"""
    
    def test_logger_instance(self):
        """测试全局 logger 实例"""
        self.assertIsInstance(logger, Logger)
    
    def test_tb_logger_instance(self):
        """测试全局 tb_logger 实例"""
        self.assertIsInstance(tb_logger, TBLogger)
    
    def test_failed_logger_instance(self):
        """测试全局 failed_logger 实例"""
        self.assertIsInstance(failed_logger, FailedLogger)


if __name__ == '__main__':
    unittest.main(verbosity=2)
