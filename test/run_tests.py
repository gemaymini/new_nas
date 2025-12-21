# -*- coding: utf-8 -*-
"""
运行所有单元测试
"""
import sys
import os
import unittest
import argparse

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保 src 在路径中
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# 切换到项目根目录，确保数据集路径正确
os.chdir(PROJECT_ROOT)


def run_all_tests(verbosity=2, pattern='test_*.py'):
    """运行所有测试"""
    # 获取测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 发现并加载测试
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_tests(test_modules, verbosity=2):
    """运行指定的测试模块"""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    for module_name in test_modules:
        try:
            # 导入测试模块
            module = __import__(module_name)
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def print_coverage_summary():
    """打印测试覆盖摘要"""
    test_modules = [
        ('test_config', '配置模块'),
        ('test_encoding', '编码模块'),
        ('test_search_space', '搜索空间模块'),
        ('test_network', '网络构建模块'),
        ('test_mutation', '变异算子模块'),
        ('test_dataset', '数据集模块'),
        ('test_trainer', '训练器模块'),
        ('test_evaluator', '评估器模块'),
        ('test_logger', '日志模块'),
        ('test_evolution', '进化算法模块'),
    ]
    
    print("\n" + "=" * 60)
    print("测试覆盖模块")
    print("=" * 60)
    for module, desc in test_modules:
        print(f"  ✓ {module}.py - {desc}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行 NAS 项目单元测试')
    parser.add_argument('-v', '--verbosity', type=int, default=2, 
                        help='输出详细程度 (0, 1, 2)')
    parser.add_argument('-m', '--module', type=str, nargs='+',
                        help='指定要运行的测试模块 (例如: test_encoding test_network)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有测试模块')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (跳过耗时的测试)')
    
    args = parser.parse_args()
    
    if args.list:
        print_coverage_summary()
        sys.exit(0)
    
    print("\n" + "=" * 60)
    print("NAS 项目单元测试")
    print("=" * 60 + "\n")
    
    if args.module:
        result = run_specific_tests(args.module, args.verbosity)
    else:
        result = run_all_tests(args.verbosity)
    
    print_coverage_summary()
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    # 如果有失败或错误，退出码为 1
    sys.exit(0 if result.wasSuccessful() else 1)
