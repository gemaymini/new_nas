# -*- coding: utf-8 -*-
"""
单元测试 - 搜索空间模块 (search_space.py)
测试 SearchSpace, PopulationInitializer 类
"""
import sys
import os
import unittest

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from core.search_space import SearchSpace, PopulationInitializer, search_space, population_initializer
from core.encoding import Encoder, Individual
from configuration.config import config


class TestSearchSpace(unittest.TestCase):
    """测试 SearchSpace 类"""
    
    def setUp(self):
        """设置测试实例"""
        self.ss = SearchSpace()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.ss.min_unit_num, config.MIN_UNIT_NUM)
        self.assertEqual(self.ss.max_unit_num, config.MAX_UNIT_NUM)
        self.assertEqual(self.ss.min_block_num, config.MIN_BLOCK_NUM)
        self.assertEqual(self.ss.max_block_num, config.MAX_BLOCK_NUM)
    
    def test_sample_unit_num(self):
        """测试采样 unit 数量"""
        for _ in range(100):
            unit_num = self.ss.sample_unit_num()
            self.assertGreaterEqual(unit_num, config.MIN_UNIT_NUM)
            self.assertLessEqual(unit_num, config.MAX_UNIT_NUM)
    
    def test_sample_block_num(self):
        """测试采样 block 数量"""
        for _ in range(100):
            block_num = self.ss.sample_block_num()
            self.assertGreaterEqual(block_num, config.MIN_BLOCK_NUM)
            self.assertLessEqual(block_num, config.MAX_BLOCK_NUM)
    
    def test_sample_channel(self):
        """测试采样通道数"""
        for _ in range(100):
            channel = self.ss.sample_channel()
            self.assertIn(channel, config.CHANNEL_OPTIONS)
    
    def test_sample_groups_no_limit(self):
        """测试采样分组数（无限制）"""
        for _ in range(100):
            groups = self.ss.sample_groups()
            self.assertIn(groups, config.GROUP_OPTIONS)
    
    def test_sample_groups_with_limit(self):
        """测试采样分组数（有限制）"""
        max_groups = 8
        for _ in range(100):
            groups = self.ss.sample_groups(max_groups)
            self.assertLessEqual(groups, max_groups)
            self.assertIn(groups, config.GROUP_OPTIONS)
    
    def test_sample_groups_with_small_limit(self):
        """测试采样分组数（小限制）"""
        max_groups = 1
        for _ in range(50):
            groups = self.ss.sample_groups(max_groups)
            self.assertEqual(groups, 1)
    
    def test_sample_pool_type(self):
        """测试采样池化类型"""
        for _ in range(100):
            pool_type = self.ss.sample_pool_type()
            self.assertIn(pool_type, config.POOL_TYPE_OPTIONS)
    
    def test_sample_pool_stride(self):
        """测试采样池化步长"""
        for _ in range(100):
            pool_stride = self.ss.sample_pool_stride()
            self.assertIn(pool_stride, config.POOL_STRIDE_OPTIONS)
    
    def test_sample_senet(self):
        """测试采样 SENet 开关"""
        for _ in range(100):
            senet = self.ss.sample_senet()
            self.assertIn(senet, config.SENET_OPTIONS)
    
    def test_sample_block_params(self):
        """测试采样完整的 block 参数"""
        for _ in range(50):
            bp = self.ss.sample_block_params()
            
            self.assertIn(bp.out_channels, config.CHANNEL_OPTIONS)
            self.assertIn(bp.groups, config.GROUP_OPTIONS)
            self.assertLessEqual(bp.groups, bp.out_channels)
            self.assertIn(bp.pool_type, config.POOL_TYPE_OPTIONS)
            self.assertIn(bp.pool_stride, config.POOL_STRIDE_OPTIONS)
            self.assertIn(bp.has_senet, config.SENET_OPTIONS)


class TestPopulationInitializer(unittest.TestCase):
    """测试 PopulationInitializer 类"""
    
    def setUp(self):
        """设置测试实例"""
        self.ss = SearchSpace()
        self.pi = PopulationInitializer(self.ss)
    
    def test_create_valid_individual(self):
        """测试创建有效个体"""
        ind = self.pi.create_valid_individual()
        
        self.assertIsNotNone(ind)
        self.assertIsInstance(ind, Individual)
        self.assertIsInstance(ind.encoding, list)
        self.assertGreater(len(ind.encoding), 0)
        
        # 验证编码有效性
        self.assertTrue(Encoder.validate_encoding(ind.encoding))
    
    def test_create_multiple_valid_individuals(self):
        """测试创建多个有效个体"""
        individuals = []
        for _ in range(10):
            ind = self.pi.create_valid_individual()
            self.assertIsNotNone(ind)
            self.assertTrue(Encoder.validate_encoding(ind.encoding))
            individuals.append(ind)
        
        # 检查编码多样性
        encodings = [str(ind.encoding) for ind in individuals]
        unique_encodings = set(encodings)
        # 至少应该有一些不同的编码（概率上几乎一定）
        self.assertGreater(len(unique_encodings), 1)
    
    def test_create_constrained_encoding(self):
        """测试创建约束编码"""
        encoding = self.pi._create_constrained_encoding()
        
        self.assertIsInstance(encoding, list)
        self.assertGreater(len(encoding), 0)
        
        # 解码并检查结构
        unit_num = encoding[0]
        self.assertGreaterEqual(unit_num, config.MIN_UNIT_NUM)
        self.assertLessEqual(unit_num, config.MAX_UNIT_NUM)
    
    def test_constrained_encoding_respects_downsampling(self):
        """测试约束编码遵守下采样限制"""
        # 创建多个编码，检查下采样次数
        for _ in range(20):
            encoding = self.pi._create_constrained_encoding()
            
            if Encoder.validate_encoding(encoding):
                # 统计 stride=2 的次数
                _, _, block_params_list = Encoder.decode(encoding)
                downsample_count = sum(
                    1 for unit in block_params_list 
                    for bp in unit if bp.pool_stride == 2
                )
                
                max_ds = Encoder.get_max_downsampling()
                self.assertLessEqual(downsample_count, max_ds)


class TestGlobalInstances(unittest.TestCase):
    """测试全局实例"""
    
    def test_search_space_instance(self):
        """测试全局 search_space 实例"""
        self.assertIsInstance(search_space, SearchSpace)
    
    def test_population_initializer_instance(self):
        """测试全局 population_initializer 实例"""
        self.assertIsInstance(population_initializer, PopulationInitializer)
    
    def test_global_instance_functionality(self):
        """测试全局实例功能"""
        ind = population_initializer.create_valid_individual()
        self.assertIsNotNone(ind)
        self.assertTrue(Encoder.validate_encoding(ind.encoding))


if __name__ == '__main__':
    unittest.main(verbosity=2)
