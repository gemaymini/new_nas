# -*- coding: utf-8 -*-
"""
单元测试 - 变异算子模块 (mutation.py)
测试 MutationOperator, SelectionOperator, CrossoverOperator 类
"""
import sys
import os
import unittest
import copy

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from search.mutation import MutationOperator, SelectionOperator, CrossoverOperator
from search.mutation import mutation_operator, selection_operator, crossover_operator
from core.encoding import Encoder, Individual
from configuration.config import config


class TestMutationOperator(unittest.TestCase):
    """测试 MutationOperator 类"""
    
    def setUp(self):
        """设置测试用例"""
        self.mo = MutationOperator()
        # 创建一个有效的编码: 3个unit, 每个2个block
        self.valid_encoding = [
            3,          # unit_num
            2, 2, 2,    # block_nums
            # Unit 1
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            # Unit 2
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            # Unit 3
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        self.valid_individual = Individual(self.valid_encoding)
        self.valid_individual.id = 1
    
    def test_swap_blocks(self):
        """测试交换块"""
        original = copy.deepcopy(self.valid_encoding)
        result = self.mo.swap_blocks(self.valid_encoding)
        
        self.assertIsInstance(result, list)
        # 结构应该保持不变
        self.assertEqual(result[0], original[0])  # unit_num
        self.assertEqual(result[1:4], original[1:4])  # block_nums
    
    def test_swap_units(self):
        """测试交换单元"""
        original = copy.deepcopy(self.valid_encoding)
        result = self.mo.swap_units(self.valid_encoding)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], original[0])  # unit_num 不变
        # 总 block 数应该不变
        total_blocks = sum(result[1:4])
        self.assertEqual(total_blocks, 6)
    
    def test_swap_units_single_unit(self):
        """测试单 unit 时交换单元"""
        # 只有一个 unit 的编码
        single_unit_encoding = [
            1, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
        ]
        result = self.mo.swap_units(single_unit_encoding)
        # 无法交换，返回原编码
        self.assertEqual(result, single_unit_encoding)
    
    def test_add_unit(self):
        """测试添加单元"""
        # 使用较少 unit 的编码
        encoding = [
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ]
        
        result = self.mo.add_unit(encoding)
        
        # 如果成功添加，unit 数增加
        if result[0] > encoding[0]:
            self.assertEqual(result[0], encoding[0] + 1)
    
    def test_add_unit_max_reached(self):
        """测试达到最大 unit 数时添加单元"""
        # 创建最大 unit 数的编码
        max_units = config.MAX_UNIT_NUM
        encoding = [max_units] + [config.MIN_BLOCK_NUM] * max_units
        for _ in range(max_units * config.MIN_BLOCK_NUM):
            encoding.extend([16, 2, 0, 1, 0])
        
        result = self.mo.add_unit(encoding)
        # 无法添加，返回原编码
        self.assertEqual(result[0], max_units)
    
    def test_add_block(self):
        """测试添加块"""
        result = self.mo.add_block(self.valid_encoding)
        
        self.assertIsInstance(result, list)
        # 总 block 数可能增加
        new_total = sum(result[1:result[0]+1])
        old_total = sum(self.valid_encoding[1:4])
        self.assertGreaterEqual(new_total, old_total)
    
    def test_delete_unit(self):
        """测试删除单元"""
        result = self.mo.delete_unit(self.valid_encoding)
        
        # 如果成功删除，unit 数减少
        if result[0] < self.valid_encoding[0]:
            self.assertEqual(result[0], self.valid_encoding[0] - 1)
    
    def test_delete_unit_min_reached(self):
        """测试达到最小 unit 数时删除单元"""
        # 创建最小 unit 数的编码
        min_units = config.MIN_UNIT_NUM
        encoding = [min_units] + [config.MIN_BLOCK_NUM] * min_units
        for _ in range(min_units * config.MIN_BLOCK_NUM):
            encoding.extend([16, 2, 0, 1, 0])
        
        result = self.mo.delete_unit(encoding)
        # 无法删除，返回原编码
        self.assertEqual(result[0], min_units)
    
    def test_delete_block(self):
        """测试删除块"""
        result = self.mo.delete_block(self.valid_encoding)
        
        self.assertIsInstance(result, list)
        # 总 block 数可能减少
        new_total = sum(result[1:result[0]+1])
        old_total = sum(self.valid_encoding[1:4])
        self.assertLessEqual(new_total, old_total)
    
    def test_modify_block(self):
        """测试修改块"""
        result = self.mo.modify_block(self.valid_encoding)
        
        self.assertIsInstance(result, list)
        # 结构应该保持不变
        self.assertEqual(result[0], self.valid_encoding[0])
    
    def test_mutate(self):
        """测试完整变异操作"""
        result = self.mo.mutate(self.valid_individual)
        
        self.assertIsInstance(result, Individual)
        self.assertIsInstance(result.encoding, list)
        self.assertGreater(len(result.encoding), 0)


class TestSelectionOperator(unittest.TestCase):
    """测试 SelectionOperator 类"""
    
    def setUp(self):
        """设置测试用例"""
        self.so = SelectionOperator()
        # 创建测试种群
        self.population = []
        for i in range(10):
            ind = Individual([3, 2, 2, 2] + [16, 2, 0, 1, 0] * 6)
            ind.id = i
            ind.fitness = i * 0.1  # 0.0 到 0.9
            self.population.append(ind)
    
    def test_tournament_selection(self):
        """测试锦标赛选择"""
        winners = self.so.tournament_selection(self.population)
        
        self.assertIsInstance(winners, list)
        self.assertEqual(len(winners), config.TOURNAMENT_WINNERS)
        
        for winner in winners:
            self.assertIsInstance(winner, Individual)
    
    def test_tournament_selection_custom_size(self):
        """测试自定义锦标赛大小"""
        winners = self.so.tournament_selection(
            self.population, 
            tournament_size=3, 
            num_winners=1
        )
        
        self.assertEqual(len(winners), 1)
    
    def test_tournament_selection_small_population(self):
        """测试小种群"""
        small_pop = self.population[:3]
        winners = self.so.tournament_selection(
            small_pop, 
            tournament_size=5,  # 大于种群大小
            num_winners=2
        )
        
        self.assertEqual(len(winners), 2)
    
    def test_tournament_selection_returns_best(self):
        """测试锦标赛选择返回最佳个体"""
        # 运行多次，最佳个体应该经常被选中
        best_count = 0
        for _ in range(100):
            winners = self.so.tournament_selection(self.population)
            if any(w.id == 9 for w in winners):  # 最佳个体 id=9, fitness=0.9
                best_count += 1
        
        # 最佳个体应该在大多数情况下被选中
        self.assertGreater(best_count, 30)


class TestCrossoverOperator(unittest.TestCase):
    """测试 CrossoverOperator 类"""
    
    def setUp(self):
        """设置测试用例"""
        self.co = CrossoverOperator()
        
        # 父代1: 3个unit
        self.parent1 = Individual([
            3, 2, 2, 2,
            16, 2, 0, 1, 0,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 1,
            32, 4, 1, 1, 0,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ])
        self.parent1.id = 1
        self.parent1.fitness = 0.8
        
        # 父代2: 4个unit
        self.parent2 = Individual([
            4, 2, 2, 2, 2,
            8, 1, 0, 1, 0,
            8, 1, 1, 1, 0,
            16, 2, 0, 1, 1,
            16, 2, 1, 1, 0,
            32, 4, 0, 1, 0,
            32, 4, 1, 1, 1,
            64, 8, 0, 1, 0,
            64, 8, 1, 1, 1,
        ])
        self.parent2.id = 2
        self.parent2.fitness = 0.7
    
    def test_uniform_unit_crossover(self):
        """测试均匀单元交叉"""
        child1, child2 = self.co.uniform_unit_crossover(self.parent1, self.parent2)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
        
        # 子代应该有有效的编码
        self.assertGreater(len(child1.encoding), 0)
        self.assertGreater(len(child2.encoding), 0)
        
        # 检查 unit 数在合理范围内
        self.assertGreaterEqual(child1.encoding[0], config.MIN_UNIT_NUM)
        self.assertLessEqual(child1.encoding[0], config.MAX_UNIT_NUM)
    
    def test_crossover(self):
        """测试交叉操作"""
        child1, child2 = self.co.crossover(self.parent1, self.parent2)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
    
    def test_crossover_same_parents(self):
        """测试相同父代交叉"""
        child1, child2 = self.co.crossover(self.parent1, self.parent1)
        
        # 应该仍然能正常工作
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)


class TestGlobalOperatorInstances(unittest.TestCase):
    """测试全局算子实例"""
    
    def test_mutation_operator_instance(self):
        """测试全局 mutation_operator 实例"""
        self.assertIsInstance(mutation_operator, MutationOperator)
    
    def test_selection_operator_instance(self):
        """测试全局 selection_operator 实例"""
        self.assertIsInstance(selection_operator, SelectionOperator)
    
    def test_crossover_operator_instance(self):
        """测试全局 crossover_operator 实例"""
        self.assertIsInstance(crossover_operator, CrossoverOperator)


if __name__ == '__main__':
    unittest.main(verbosity=2)
