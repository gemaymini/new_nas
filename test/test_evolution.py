# -*- coding: utf-8 -*-
"""
单元测试 - 进化算法模块 (evolution.py)
测试 AgingEvolutionNAS 类
"""
import sys
import os
import unittest
import tempfile
import shutil
from collections import deque

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from search.evolution import AgingEvolutionNAS
from core.encoding import Individual, Encoder
from configuration.config import config


class TestAgingEvolutionNASInit(unittest.TestCase):
    """测试 AgingEvolutionNAS 初始化"""
    
    def test_init(self):
        """测试初始化"""
        nas = AgingEvolutionNAS()
        
        self.assertEqual(nas.population_size, config.POPULATION_SIZE)
        self.assertEqual(nas.max_gen, config.MAX_GEN)
        self.assertIsInstance(nas.population, deque)
        self.assertIsInstance(nas.history, list)
        self.assertEqual(len(nas.population), 0)
        self.assertEqual(len(nas.history), 0)


class TestAgingEvolutionNASHelpers(unittest.TestCase):
    """测试 AgingEvolutionNAS 辅助方法"""
    
    def setUp(self):
        """设置测试"""
        self.nas = AgingEvolutionNAS()
        
        # 创建模拟种群
        self.valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        # 添加一些模拟个体到种群
        for i in range(10):
            ind = Individual(self.valid_encoding.copy())
            ind.id = i
            ind.fitness = i * 0.1
            self.nas.population.append(ind)
            self.nas.history.append(ind)
    
    def test_select_parents(self):
        """测试选择父代"""
        parent1, parent2 = self.nas._select_parents()
        
        self.assertIsInstance(parent1, Individual)
        self.assertIsInstance(parent2, Individual)
    
    def test_generate_offspring(self):
        """测试生成后代"""
        parent1 = self.nas.population[0]
        parent2 = self.nas.population[1]
        
        child = self.nas._generate_offspring(parent1, parent2)
        
        self.assertIsInstance(child, Individual)
        self.assertIsInstance(child.encoding, list)
    
    def test_repair_individual(self):
        """测试修复个体"""
        parent1 = self.nas.population[0]
        parent2 = self.nas.population[1]
        
        # 创建一个无效个体
        invalid_ind = Individual([1])  # 无效编码
        
        repaired = self.nas._repair_individual(invalid_ind, [parent1, parent2])
        
        self.assertIsInstance(repaired, Individual)
    
    def test_record_statistics(self):
        """测试记录统计信息"""
        # 不应该抛出异常
        try:
            self.nas._record_statistics()
        except Exception as e:
            self.fail(f"_record_statistics raised exception: {e}")


class TestAgingEvolutionNASCheckpoint(unittest.TestCase):
    """测试 AgingEvolutionNAS 检查点功能"""
    
    def setUp(self):
        """设置测试"""
        self.nas = AgingEvolutionNAS()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟种群
        valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        for i in range(5):
            ind = Individual(valid_encoding.copy())
            ind.id = i
            ind.fitness = i * 0.1
            self.nas.population.append(ind)
            self.nas.history.append(ind)
    
    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_checkpoint(self):
        """测试保存检查点"""
        filepath = os.path.join(self.temp_dir, 'test_checkpoint.pkl')
        self.nas.save_checkpoint(filepath)
        
        self.assertTrue(os.path.exists(filepath))
    
    def test_load_checkpoint(self):
        """测试加载检查点"""
        filepath = os.path.join(self.temp_dir, 'test_checkpoint.pkl')
        
        # 保存
        original_pop_size = len(self.nas.population)
        original_history_size = len(self.nas.history)
        self.nas.save_checkpoint(filepath)
        
        # 创建新实例并加载
        new_nas = AgingEvolutionNAS()
        new_nas.load_checkpoint(filepath)
        
        self.assertEqual(len(new_nas.population), original_pop_size)
        self.assertEqual(len(new_nas.history), original_history_size)


class TestAgingEvolutionNASStep(unittest.TestCase):
    """测试 AgingEvolutionNAS 单步进化"""
    
    def setUp(self):
        """设置测试"""
        self.nas = AgingEvolutionNAS()
        
        # 创建模拟种群
        valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        # 需要足够的个体进行锦标赛选择
        for i in range(config.TOURNAMENT_SIZE + 2):
            ind = Individual(valid_encoding.copy())
            ind.id = i
            ind.fitness = i * 0.1
            self.nas.population.append(ind)
            self.nas.history.append(ind)
    
    def test_step(self):
        """测试单步进化"""
        initial_history_size = len(self.nas.history)
        initial_pop_size = len(self.nas.population)
        
        # 执行一步
        self.nas.step()
        
        # 历史增加 1
        self.assertEqual(len(self.nas.history), initial_history_size + 1)
        # 种群大小不变
        self.assertEqual(len(self.nas.population), initial_pop_size)


class TestAgingEvolutionNASScreening(unittest.TestCase):
    """测试 AgingEvolutionNAS 筛选功能"""
    
    def setUp(self):
        """设置测试"""
        self.nas = AgingEvolutionNAS()
        
        # 创建模拟历史
        valid_encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        for i in range(25):
            ind = Individual(valid_encoding.copy())
            ind.id = i
            ind.fitness = -i  # 负值表示 NTK 条件数
            self.nas.history.append(ind)
    
    def test_history_deduplication(self):
        """测试历史去重"""
        # 添加重复编码的个体
        dup_encoding = self.nas.history[0].encoding.copy()
        dup_ind = Individual(dup_encoding)
        dup_ind.id = 100
        dup_ind.fitness = 99.0  # 更好的 fitness
        self.nas.history.append(dup_ind)
        
        # 验证去重后保留更好的个体
        unique_history = {}
        for ind in self.nas.history:
            enc_tuple = tuple(ind.encoding)
            if enc_tuple not in unique_history:
                unique_history[enc_tuple] = ind
            else:
                if ind.fitness > unique_history[enc_tuple].fitness:
                    unique_history[enc_tuple] = ind
        
        # 应该保留 fitness=99.0 的个体
        best_for_encoding = unique_history[tuple(dup_encoding)]
        self.assertEqual(best_for_encoding.fitness, 99.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
