# -*- coding: utf-8 -*-
"""
操作记录分析示例
展示如何使用操作分析工具
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply.analyze_operations import OperationsAnalyzer

def demo_analysis():
    """演示分析功能"""
    print("=== 变异和交叉操作分析演示 ===\\n")
    
    # 创建分析器
    analyzer = OperationsAnalyzer()
    
    if not analyzer.operations:
        print("没有找到操作记录，请先运行一些搜索实验。")
        return
    
    # 1. 生成基本报告
    print("1. 生成基本分析报告:")
    report = analyzer.generate_report()
    print(report)
    print("\\n" + "="*60 + "\\n")
    
    # 2. 详细的变异频率分析
    print("2. 详细变异频率分析:")
    mutation_freq = analyzer.analyze_mutation_frequency()
    if mutation_freq:
        total_mutations = sum(mutation_freq.values())
        print(f"总变异次数: {total_mutations}")
        for mut_type, count in sorted(mutation_freq.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_mutations) * 100
            print(f"  {mut_type}: {count} 次 ({percentage:.1f}%)")
    else:
        print("  没有变异记录")
    print("\\n" + "="*60 + "\\n")
    
    # 3. 结构变化分析
    print("3. 结构变化详细分析:")
    structure_changes = analyzer.analyze_structure_changes()
    
    print("Unit数量变化:")
    unit_dist = structure_changes['unit_changes']['distribution']
    for change, count in sorted(unit_dist.items()):
        if change == 0:
            print(f"  无变化: {count} 次")
        elif change > 0:
            print(f"  增加{change}个unit: {count} 次")
        else:
            print(f"  减少{abs(change)}个unit: {count} 次")
    
    print("\\nBlock数量变化:")
    block_dist = structure_changes['block_changes']['distribution']
    for change, count in sorted(block_dist.items()):
        if change == 0:
            print(f"  无变化: {count} 次")
        elif change > 0:
            print(f"  增加{change}个block: {count} 次")
        else:
            print(f"  减少{abs(change)}个block: {count} 次")
    print("\\n" + "="*60 + "\\n")
    
    # 4. 交叉模式分析
    print("4. 交叉模式分析:")
    crossover_patterns = analyzer.analyze_crossover_patterns()
    if crossover_patterns:
        print(f"总交叉次数: {crossover_patterns['total_crossovers']}")
        
        unit_dist = crossover_patterns['unit_num_distribution']
        print("\\n生成的子个体unit数量分布:")
        for unit_num, count in sorted(unit_dist.items()):
            print(f"  {unit_num} units: {count} 次")
        
        gen_units_dist = crossover_patterns['generated_units']['distribution']
        print("\\n每次交叉生成新unit的数量:")
        for gen_count, freq in sorted(gen_units_dist.items()):
            print(f"  生成{gen_count}个新unit: {freq} 次")
        
        selection_pattern = crossover_patterns['selection_pattern']
        if selection_pattern:
            total_selections = sum(selection_pattern.values())
            parent1_selections = selection_pattern.get('from_parent1', 0)
            parent2_selections = selection_pattern.get('from_parent2', 0)
            print(f"\\n选择偏好:")
            print(f"  从parent1选择: {parent1_selections} 次 ({parent1_selections/total_selections*100:.1f}%)")
            print(f"  从parent2选择: {parent2_selections} 次 ({parent2_selections/total_selections*100:.1f}%)")
    else:
        print("  没有交叉记录")
    
    print("\\n" + "="*60)
    print("分析完成！")
    print("\\n可以使用以下方法获得更详细的分析:")
    print("- analyzer.plot_mutation_frequency() # 绘制变异频率图")
    print("- analyzer.export_to_csv('path.csv') # 导出到CSV")


if __name__ == '__main__':
    demo_analysis()