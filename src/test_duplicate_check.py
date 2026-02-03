# -*- coding: utf-8 -*-
"""
测试个体判重功能
"""
import unittest
from core.encoding import Individual, DuplicateChecker
from configuration.config import config
class TestDuplicateCheck(unittest.TestCase):
    """测试个体判重功能的单元测试"""

    def test_duplicate_check(self):
        """测试判重功能"""
        # 创建测试个体
        ind1 = Individual()
        ind2 = Individual()
        ind3 = ind1.copy()  # 复制 ind1，编码应该相同

        # 测试相等性
        self.assertEqual(ind1, ind3)
        self.assertNotEqual(ind1, ind2)

        # 测试哈希
        self.assertEqual(hash(ind1), hash(ind3))
        self.assertNotEqual(hash(ind1), hash(ind2))

        # 测试判重函数
        population = [ind1, ind2]
        self.assertTrue(DuplicateChecker.is_duplicate(ind3, population))

        # 创建新个体并测试非重复
        ind4 = Individual()
        self.assertFalse(DuplicateChecker.is_duplicate(ind4, population))

        # 测试快速判重
        encoding_set = DuplicateChecker.get_encoding_set(population)
        self.assertTrue(DuplicateChecker.is_duplicate_fast(ind3, encoding_set))
        self.assertFalse(DuplicateChecker.is_duplicate_fast(ind4, encoding_set))

        # 测试禁用判重
        original_setting = config.ENABLE_DUPLICATE_CHECK
        try:
            config.ENABLE_DUPLICATE_CHECK = False
            self.assertFalse(DuplicateChecker.is_duplicate(ind3, population))
        finally:
            config.ENABLE_DUPLICATE_CHECK = original_setting


if __name__ == "__main__":
    unittest.main()
