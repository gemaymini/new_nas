# -*- coding: utf-8 -*-
"""
单元测试 - 配置模块 (config.py)
测试 Config 类
"""
import sys
import os
import unittest

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from configuration.config import Config, config


class TestConfig(unittest.TestCase):
    """测试 Config 类"""
    
    def test_config_instance(self):
        """测试全局配置实例"""
        self.assertIsInstance(config, Config)
    
    def test_evolution_params(self):
        """测试进化算法参数"""
        self.assertIsInstance(config.POPULATION_SIZE, int)
        self.assertGreater(config.POPULATION_SIZE, 0)
        
        self.assertIsInstance(config.MAX_GEN, int)
        self.assertGreater(config.MAX_GEN, 0)
        
        self.assertIsInstance(config.TOURNAMENT_SIZE, int)
        self.assertGreater(config.TOURNAMENT_SIZE, 0)
        
        self.assertIsInstance(config.TOURNAMENT_WINNERS, int)
        self.assertGreater(config.TOURNAMENT_WINNERS, 0)
    
    def test_screening_params(self):
        """测试筛选参数"""
        self.assertIsInstance(config.HISTORY_TOP_N1, int)
        self.assertGreater(config.HISTORY_TOP_N1, 0)
        
        self.assertIsInstance(config.SHORT_TRAIN_EPOCHS, int)
        self.assertGreater(config.SHORT_TRAIN_EPOCHS, 0)
        
        self.assertIsInstance(config.HISTORY_TOP_N2, int)
        self.assertGreater(config.HISTORY_TOP_N2, 0)
        
        self.assertIsInstance(config.FULL_TRAIN_EPOCHS, int)
        self.assertGreater(config.FULL_TRAIN_EPOCHS, 0)
    
    def test_crossover_mutation_probs(self):
        """测试交叉和变异概率"""
        self.assertIsInstance(config.PROB_CROSSOVER, float)
        self.assertGreaterEqual(config.PROB_CROSSOVER, 0)
        self.assertLessEqual(config.PROB_CROSSOVER, 1)
        
        self.assertIsInstance(config.PROB_MUTATION, float)
        self.assertGreaterEqual(config.PROB_MUTATION, 0)
        self.assertLessEqual(config.PROB_MUTATION, 1)
    
    def test_search_space_params(self):
        """测试搜索空间参数"""
        self.assertIsInstance(config.MIN_UNIT_NUM, int)
        self.assertIsInstance(config.MAX_UNIT_NUM, int)
        self.assertGreater(config.MIN_UNIT_NUM, 0)
        self.assertGreaterEqual(config.MAX_UNIT_NUM, config.MIN_UNIT_NUM)
        
        self.assertIsInstance(config.MIN_BLOCK_NUM, int)
        self.assertIsInstance(config.MAX_BLOCK_NUM, int)
        self.assertGreater(config.MIN_BLOCK_NUM, 0)
        self.assertGreaterEqual(config.MAX_BLOCK_NUM, config.MIN_BLOCK_NUM)
        
        self.assertIsInstance(config.CHANNEL_OPTIONS, list)
        self.assertGreater(len(config.CHANNEL_OPTIONS), 0)
        for ch in config.CHANNEL_OPTIONS:
            self.assertIsInstance(ch, int)
            self.assertGreater(ch, 0)
        
        self.assertIsInstance(config.GROUP_OPTIONS, list)
        self.assertGreater(len(config.GROUP_OPTIONS), 0)
        
        self.assertIsInstance(config.POOL_TYPE_OPTIONS, list)
        self.assertEqual(len(config.POOL_TYPE_OPTIONS), 2)
        self.assertIn(0, config.POOL_TYPE_OPTIONS)
        self.assertIn(1, config.POOL_TYPE_OPTIONS)
        
        self.assertIsInstance(config.POOL_STRIDE_OPTIONS, list)
        self.assertEqual(len(config.POOL_STRIDE_OPTIONS), 2)
        self.assertIn(1, config.POOL_STRIDE_OPTIONS)
        self.assertIn(2, config.POOL_STRIDE_OPTIONS)
        
        self.assertIsInstance(config.SENET_OPTIONS, list)
        self.assertEqual(len(config.SENET_OPTIONS), 2)
        self.assertIn(0, config.SENET_OPTIONS)
        self.assertIn(1, config.SENET_OPTIONS)
    
    def test_init_conv_params(self):
        """测试初始卷积参数"""
        self.assertIsInstance(config.INIT_CONV_OUT_CHANNELS, int)
        self.assertGreater(config.INIT_CONV_OUT_CHANNELS, 0)
        
        self.assertIsInstance(config.INIT_CONV_KERNEL_SIZE, int)
        self.assertGreater(config.INIT_CONV_KERNEL_SIZE, 0)
        
        self.assertIsInstance(config.INIT_CONV_STRIDE, int)
        self.assertGreater(config.INIT_CONV_STRIDE, 0)
        
        self.assertIsInstance(config.INIT_CONV_PADDING, int)
        self.assertGreaterEqual(config.INIT_CONV_PADDING, 0)
    
    def test_mutation_probs(self):
        """测试变异概率参数"""
        probs = [
            config.PROB_SWAP_BLOCKS,
            config.PROB_SWAP_UNITS,
            config.PROB_ADD_UNIT,
            config.PROB_ADD_BLOCK,
            config.PROB_DELETE_UNIT,
            config.PROB_DELETE_BLOCK,
            config.PROB_MODIFY_BLOCK,
        ]
        
        for prob in probs:
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_ntk_params(self):
        """测试 NTK 评估参数"""
        self.assertIsInstance(config.NTK_BATCH_SIZE, int)
        self.assertGreater(config.NTK_BATCH_SIZE, 0)
        
        self.assertIsInstance(config.NTK_INPUT_SIZE, tuple)
        self.assertEqual(len(config.NTK_INPUT_SIZE), 3)
        
        self.assertIsInstance(config.NTK_NUM_CLASSES, int)
        self.assertGreater(config.NTK_NUM_CLASSES, 0)
        
        self.assertIsInstance(config.NTK_PARAM_THRESHOLD, int)
        self.assertGreater(config.NTK_PARAM_THRESHOLD, 0)
    
    def test_training_params(self):
        """测试训练参数"""
        self.assertIn(config.DEVICE, ['cuda', 'cpu'])
        
        self.assertIsInstance(config.BATCH_SIZE, int)
        self.assertGreater(config.BATCH_SIZE, 0)
        
        self.assertIsInstance(config.LEARNING_RATE, float)
        self.assertGreater(config.LEARNING_RATE, 0)
        
        self.assertIsInstance(config.MOMENTUM, float)
        self.assertGreaterEqual(config.MOMENTUM, 0)
        self.assertLessEqual(config.MOMENTUM, 1)
        
        self.assertIsInstance(config.WEIGHT_DECAY, float)
        self.assertGreaterEqual(config.WEIGHT_DECAY, 0)
    
    def test_senet_params(self):
        """测试 SENet 参数"""
        self.assertIsInstance(config.SENET_REDUCTION, int)
        self.assertGreater(config.SENET_REDUCTION, 0)
    
    def test_log_params(self):
        """测试日志参数"""
        self.assertIsInstance(config.LOG_DIR, str)
        self.assertIsInstance(config.LOG_LEVEL, str)
        self.assertIn(config.LOG_LEVEL, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    def test_checkpoint_params(self):
        """测试检查点参数"""
        self.assertIsInstance(config.SAVE_CHECKPOINT, bool)
        self.assertIsInstance(config.CHECKPOINT_DIR, str)
    
    def test_tensorboard_params(self):
        """测试 TensorBoard 参数"""
        self.assertIsInstance(config.USE_TENSORBOARD, bool)
        self.assertIsInstance(config.TENSORBOARD_DIR, str)
    
    def test_architecture_constraints(self):
        """测试架构约束参数"""
        self.assertIsInstance(config.MIN_FEATURE_SIZE, int)
        self.assertGreater(config.MIN_FEATURE_SIZE, 0)
        
        self.assertIsInstance(config.INPUT_IMAGE_SIZE, int)
        self.assertGreater(config.INPUT_IMAGE_SIZE, 0)
    
    def test_random_seed(self):
        """测试随机种子"""
        self.assertIsInstance(config.RANDOM_SEED, int)
    
    def test_num_workers(self):
        """测试数据加载工作进程数"""
        self.assertIsInstance(config.NUM_WORKERS, int)
        self.assertGreaterEqual(config.NUM_WORKERS, 0)
    
    def test_get_search_space_summary(self):
        """测试获取搜索空间摘要"""
        summary = config.get_search_space_summary()
        self.assertIsInstance(summary, str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
