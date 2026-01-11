# -*- coding: utf-8 -*-
"""
变异和交叉操作分析工具
分析操作记录，生成统计报告
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Dict, Any
from configuration.config import config

class OperationsAnalyzer:
    """操作记录分析器"""
    
    def __init__(self, log_file_path: str = None):
        if log_file_path is None:
            log_file_path = os.path.join(config.LOG_DIR, 'operations_log.jsonl')
        self.log_file_path = log_file_path
        self.operations = []
        self.load_operations()
    
    def load_operations(self):
        """加载操作记录"""
        if not os.path.exists(self.log_file_path):
            print(f"操作记录文件不存在: {self.log_file_path}")
            return
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.operations.append(json.loads(line))
            print(f"加载了 {len(self.operations)} 条操作记录")
        except Exception as e:
            print(f"加载操作记录失败: {e}")
    
    def analyze_mutation_frequency(self) -> Dict[str, int]:
        """分析变异操作频率"""
        mutations = [op for op in self.operations if op['operation'] == 'mutation']
        mutation_counter = Counter()
        
        for mut in mutations:
            applied_mutations = mut['details'].get('applied_mutations', [])
            for mutation_type in applied_mutations:
                mutation_counter[mutation_type] += 1
        
        return dict(mutation_counter)
    
    def analyze_structure_changes(self) -> Dict[str, Any]:
        """分析结构变化统计"""
        mutations = [op for op in self.operations if op['operation'] == 'mutation']
        
        unit_changes = []
        block_changes = []
        encoding_length_changes = []
        
        for mut in mutations:
            details = mut['details']
            original = details.get('original_structure', {})
            new = details.get('new_structure', {})
            
            if 'unit_num' in original and 'unit_num' in new:
                unit_changes.append(new['unit_num'] - original['unit_num'])
            
            if 'total_blocks' in original and 'total_blocks' in new:
                block_changes.append(new['total_blocks'] - original['total_blocks'])
            
            encoding_change = details.get('encoding_length_change', 0)
            encoding_length_changes.append(encoding_change)
        
        return {
            'unit_changes': {
                'mean': sum(unit_changes) / len(unit_changes) if unit_changes else 0,
                'distribution': Counter(unit_changes)
            },
            'block_changes': {
                'mean': sum(block_changes) / len(block_changes) if block_changes else 0,
                'distribution': Counter(block_changes)
            },
            'encoding_length_changes': {
                'mean': sum(encoding_length_changes) / len(encoding_length_changes) if encoding_length_changes else 0,
                'distribution': Counter(encoding_length_changes)
            }
        }
    
    def analyze_crossover_patterns(self) -> Dict[str, Any]:
        """分析交叉模式"""
        crossovers = [op for op in self.operations if op['operation'] == 'crossover']
        
        if not crossovers:
            return {}
        
        unit_num_distributions = []
        selection_patterns = []
        generated_units_count = []
        
        for cross in crossovers:
            details = cross['details']
            crossover_info = details.get('crossover_info', {})
            
            unit_num_distributions.append(crossover_info.get('new_unit_num', 0))
            
            unit_selections = crossover_info.get('unit_selections', [])
            for selection in unit_selections:
                if selection['child1_from'] == 'parent1':
                    selection_patterns.append('from_parent1')
                else:
                    selection_patterns.append('from_parent2')
            
            generated_units_count.append(len(crossover_info.get('generated_units', [])))
        
        return {
            'total_crossovers': len(crossovers),
            'unit_num_distribution': Counter(unit_num_distributions),
            'selection_pattern': Counter(selection_patterns),
            'generated_units': {
                'mean': sum(generated_units_count) / len(generated_units_count) if generated_units_count else 0,
                'distribution': Counter(generated_units_count)
            }
        }
    
    def generate_report(self) -> str:
        """生成分析报告"""
        if not self.operations:
            return "没有操作记录可分析"
        
        mutation_freq = self.analyze_mutation_frequency()
        structure_changes = self.analyze_structure_changes()
        crossover_patterns = self.analyze_crossover_patterns()
        
        report = []
        report.append("=" * 60)
        report.append("变异和交叉操作分析报告")
        report.append("=" * 60)
        
        # 基本统计
        mutations = [op for op in self.operations if op['operation'] == 'mutation']
        crossovers = [op for op in self.operations if op['operation'] == 'crossover']
        
        report.append(f"总操作数: {len(self.operations)}")
        report.append(f"变异操作数: {len(mutations)}")
        report.append(f"交叉操作数: {len(crossovers)}")
        report.append("")
        
        # 变异频率分析
        report.append("变异操作频率:")
        for mut_type, count in sorted(mutation_freq.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {mut_type}: {count}")
        report.append("")
        
        # 结构变化分析
        report.append("结构变化统计:")
        unit_changes = structure_changes['unit_changes']
        report.append(f"  Unit数量平均变化: {unit_changes['mean']:.3f}")
        
        block_changes = structure_changes['block_changes']
        report.append(f"  Block数量平均变化: {block_changes['mean']:.3f}")
        
        encoding_changes = structure_changes['encoding_length_changes']
        report.append(f"  编码长度平均变化: {encoding_changes['mean']:.3f}")
        report.append("")
        
        # 交叉模式分析
        if crossover_patterns:
            report.append("交叉操作分析:")
            report.append(f"  总交叉次数: {crossover_patterns['total_crossovers']}")
            
            selection_pattern = crossover_patterns['selection_pattern']
            total_selections = sum(selection_pattern.values())
            if total_selections > 0:
                parent1_ratio = selection_pattern.get('from_parent1', 0) / total_selections
                report.append(f"  从parent1选择的比例: {parent1_ratio:.3f}")
            
            gen_units = crossover_patterns['generated_units']
            report.append(f"  平均生成新unit数: {gen_units['mean']:.3f}")
        
        return "\\n".join(report)
    
    def plot_mutation_frequency(self, save_path: str = None):
        """绘制变异频率图"""
        mutation_freq = self.analyze_mutation_frequency()
        
        if not mutation_freq:
            print("没有变异数据可绘制")
            return
        
        plt.figure(figsize=(12, 6))
        mutations, counts = zip(*sorted(mutation_freq.items(), key=lambda x: x[1], reverse=True))
        
        plt.bar(mutations, counts)
        plt.title('变异操作频率分布')
        plt.xlabel('变异类型')
        plt.ylabel('频率')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"变异频率图已保存到: {save_path}")
        
        plt.show()
    
    def export_to_csv(self, save_path: str):
        """导出操作记录到CSV"""
        if not self.operations:
            print("没有操作记录可导出")
            return
        
        # 扁平化操作记录
        flattened_data = []
        for op in self.operations:
            row = {
                'timestamp': op['timestamp'],
                'operation': op['operation'],
                'type': op.get('type', ''),
            }
            
            if op['operation'] == 'mutation':
                row.update({
                    'parent_id': op.get('parent_id', ''),
                    'child_id': op.get('child_id', ''),
                    'applied_mutations': ','.join(op['details'].get('applied_mutations', [])),
                    'unit_change': op['details'].get('new_structure', {}).get('unit_num', 0) - 
                                 op['details'].get('original_structure', {}).get('unit_num', 0),
                    'block_change': op['details'].get('new_structure', {}).get('total_blocks', 0) - 
                                  op['details'].get('original_structure', {}).get('total_blocks', 0),
                    'encoding_length_change': op['details'].get('encoding_length_change', 0)
                })
            elif op['operation'] == 'crossover':
                row.update({
                    'parent1_id': op.get('parent1_id', ''),
                    'parent2_id': op.get('parent2_id', ''),
                    'child1_id': op.get('child1_id', ''),
                    'child2_id': op.get('child2_id', ''),
                    'new_unit_num': op['details'].get('crossover_info', {}).get('new_unit_num', 0),
                    'generated_units': len(op['details'].get('crossover_info', {}).get('generated_units', []))
                })
            
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"操作记录已导出到: {save_path}")


def main():
    """主函数"""
    analyzer = OperationsAnalyzer()
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 绘制图表
    if analyzer.operations:
        try:
            # 确保输出目录存在
            output_dir = os.path.join(config.LOG_DIR, 'analysis')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 绘制变异频率图
            analyzer.plot_mutation_frequency(
                save_path=os.path.join(output_dir, 'mutation_frequency.png')
            )
            
            # 导出CSV
            analyzer.export_to_csv(
                save_path=os.path.join(output_dir, 'operations_export.csv')
            )
            
        except ImportError:
            print("matplotlib 或 pandas 未安装，跳过图表绘制和CSV导出")


if __name__ == '__main__':
    main()