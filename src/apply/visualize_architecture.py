# -*- coding: utf-8 -*-
"""
DARTS 网络架构层次图可视化模块

绘制完整的 DARTS 网络架构图，展示模型的层次结构：
- 整体网络结构（Stem -> Cells -> Classifier）
- Cell 结构（Normal Cell 和 Reduction Cell）
- Node 结构（节点内的操作连接）

使用方法:
1. 从 JSON 文件读取编码并绘制:
   python visualize_architecture.py --encoding_file path/to/encoding.json

2. 从命令行指定编码:
   python visualize_architecture.py --normal_cell "0,2,1,3,..." --reduction_cell "0,2,1,3,..."

3. 自定义输出:
   python visualize_architecture.py --encoding_file encoding.json --output_dir ./figs --format pdf

依赖:
- graphviz (Python 包): pip install graphviz
- Graphviz (系统软件): https://graphviz.org/download/
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Optional

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration.config import config
from core.encoding import CellEncoding, Edge

# 操作名称到显示名称的映射
OP_DISPLAY_NAMES = {
    'zero': 'zero',
    'skip_connect': 'skip',
    'sep_conv_3x3': 'sep 3×3',
    'sep_conv_5x5': 'sep 5×5',
    'dil_conv_3x3': 'dil 3×3',
    'dil_conv_5x5': 'dil 5×5',
    'max_pool_3x3': 'max 3×3',
    'avg_pool_3x3': 'avg 3×3',
}

# 颜色方案
COLORS = {
    'stem': '#3498DB',           # 蓝色 - Stem
    'normal_cell': '#27AE60',    # 绿色 - Normal Cell
    'reduction_cell': '#E74C3C', # 红色 - Reduction Cell
    'classifier': '#9B59B6',     # 紫色 - Classifier
    'node': '#F39C12',           # 橙色 - Node
    'input': '#5DADE2',          # 浅蓝 - Input
    'output': '#58D68D',         # 浅绿 - Output
    'edge': '#7F8C8D',           # 灰色 - Edge
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='可视化 DARTS 网络架构层次图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 编码输入方式
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--encoding_file', type=str,
        help='包含 Cell 编码的 JSON 文件路径'
    )
    input_group.add_argument(
        '--normal_cell', type=str,
        help='Normal Cell 编码 (逗号分隔的整数列表)'
    )
    
    parser.add_argument(
        '--reduction_cell', type=str,
        help='Reduction Cell 编码 (逗号分隔的整数列表)，与 --normal_cell 一起使用'
    )
    
    # 网络结构参数
    parser.add_argument(
        '--cells_per_stage', type=int, default=None,
        help=f'每个 Stage 的 Cell 数量 (默认: {config.CELLS_PER_STAGE})'
    )
    parser.add_argument(
        '--num_stages', type=int, default=None,
        help=f'Stage 数量 (默认: {config.NUM_STAGES})'
    )
    
    # 可视化选项
    parser.add_argument(
        '--output_dir', type=str, default='./visualizations',
        help='输出目录 (默认: ./visualizations)'
    )
    parser.add_argument(
        '--output_name', type=str, default='architecture',
        help='输出文件名前缀 (默认: architecture)'
    )
    parser.add_argument(
        '--format', type=str, default='png',
        choices=['png', 'pdf', 'svg', 'jpg'],
        help='输出格式 (默认: png)'
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='图像 DPI (默认: 300)'
    )
    parser.add_argument(
        '--show_details', action='store_true',
        help='显示详细的 Cell 内部结构'
    )
    parser.add_argument(
        '--compact', action='store_true',
        help='紧凑模式，只显示主要结构'
    )
    
    args = parser.parse_args()
    
    if args.normal_cell and not args.reduction_cell:
        parser.error("--reduction_cell 必须与 --normal_cell 一起指定")
    
    return args


def parse_cell_encoding(encoding_str: str) -> List[int]:
    """解析 Cell 编码字符串"""
    try:
        encoding = [int(x.strip()) for x in encoding_str.split(',')]
        return encoding
    except ValueError as e:
        raise ValueError(f"无法解析编码字符串: {encoding_str}. 错误: {e}")


def load_encoding_from_file(filepath: str) -> Tuple[List[int], List[int]]:
    """从 JSON 文件加载 Cell 编码"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"编码文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'normal_cell' not in data:
        raise KeyError("JSON 文件中缺少 'normal_cell' 字段")
    if 'reduction_cell' not in data:
        raise KeyError("JSON 文件中缺少 'reduction_cell' 字段")
    
    return data['normal_cell'], data['reduction_cell']


def create_network_overview(cells_per_stage: int, num_stages: int, 
                           output_format: str = 'png', dpi: int = 300):
    """
    创建网络整体结构概览图
    
    展示: Input -> Stem -> [Normal Cells -> Reduction Cell] x N -> Classifier -> Output
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装 graphviz 包。请运行: pip install graphviz")
    
    dot = Digraph(
        name='DARTS_Network_Overview',
        format=output_format,
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.4',
            'ranksep': '0.6',
            'fontname': 'Arial',
            'fontsize': '14',
            'label': 'DARTS Network Architecture Overview',
            'labelloc': 't',
            'labeljust': 'c',
            'bgcolor': 'white',
            'dpi': str(dpi),
        },
        node_attr={
            'fontname': 'Arial',
            'fontsize': '11',
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '9',
        }
    )
    
    # Input
    dot.node('input', 'Input\n(3×32×32)', 
            shape='box', style='rounded,filled',
            fillcolor='#E8F4FD', color=COLORS['input'], penwidth='2')
    
    # Stem
    dot.node('stem', 'Stem\n(Conv 3×3)', 
            shape='box', style='rounded,filled',
            fillcolor='#D6EAF8', color=COLORS['stem'], penwidth='2')
    
    dot.edge('input', 'stem', penwidth='2')
    
    prev_node = 'stem'
    cell_idx = 0
    
    for stage_idx in range(num_stages):
        # Reduction Cell (除了第一个 Stage)
        if stage_idx > 0:
            cell_name = f'reduction_{stage_idx}'
            dot.node(cell_name, f'Reduction Cell {stage_idx}\n(stride=2)',
                    shape='box', style='rounded,filled',
                    fillcolor='#FADBD8', color=COLORS['reduction_cell'], penwidth='2')
            dot.edge(prev_node, cell_name, penwidth='2')
            prev_node = cell_name
            cell_idx += 1
        
        # Normal Cells (使用子图分组)
        with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage:
            stage.attr(
                label=f'Stage {stage_idx + 1}',
                style='rounded,dashed',
                color='#BDC3C7',
                fontcolor='#7F8C8D',
            )
            
            for i in range(cells_per_stage):
                cell_name = f'normal_{stage_idx}_{i}'
                stage.node(cell_name, f'Normal Cell {cell_idx}',
                          shape='box', style='rounded,filled',
                          fillcolor='#D5F5E3', color=COLORS['normal_cell'], penwidth='2')
                dot.edge(prev_node, cell_name, penwidth='2')
                prev_node = cell_name
                cell_idx += 1
    
    # Global Average Pooling
    dot.node('gap', 'Global Avg Pool', 
            shape='box', style='rounded,filled',
            fillcolor='#E8DAEF', color=COLORS['classifier'], penwidth='2')
    dot.edge(prev_node, 'gap', penwidth='2')
    
    # Classifier
    dot.node('classifier', 'Classifier\n(FC → Softmax)', 
            shape='box', style='rounded,filled',
            fillcolor='#E8DAEF', color=COLORS['classifier'], penwidth='2')
    dot.edge('gap', 'classifier', penwidth='2')
    
    # Output
    dot.node('output', 'Output\n(num_classes)', 
            shape='box', style='rounded,filled',
            fillcolor='#D5F5E3', color=COLORS['output'], penwidth='2')
    dot.edge('classifier', 'output', penwidth='2')
    
    return dot


def create_cell_detail_diagram(cell_encoding: CellEncoding, cell_type: str,
                               output_format: str = 'png', dpi: int = 300):
    """
    创建 Cell 详细结构图
    
    展示 Cell 内部的 DAG 结构
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装 graphviz 包。请运行: pip install graphviz")
    
    color = COLORS['normal_cell'] if cell_type == 'Normal' else COLORS['reduction_cell']
    bg_color = '#D5F5E3' if cell_type == 'Normal' else '#FADBD8'
    
    dot = Digraph(
        name=f'{cell_type}_Cell_Detail',
        format=output_format,
        graph_attr={
            'rankdir': 'LR',
            'splines': 'polyline',
            'nodesep': '0.5',
            'ranksep': '1.0',
            'fontname': 'Arial',
            'fontsize': '14',
            'label': f'{cell_type} Cell Structure\n(Nodes connected by operations)',
            'labelloc': 't',
            'labeljust': 'c',
            'bgcolor': 'white',
            'dpi': str(dpi),
        },
    )
    
    # 输入节点
    with dot.subgraph(name='cluster_inputs') as inputs:
        inputs.attr(label='Inputs', style='rounded,dashed', color='#BDC3C7')
        inputs.node('c_k_2', 'c_{k-2}\n(prev-prev output)',
                   shape='box', style='rounded,filled',
                   fillcolor='#D6EAF8', color=COLORS['input'], penwidth='2')
        inputs.node('c_k_1', 'c_{k-1}\n(prev output)',
                   shape='box', style='rounded,filled',
                   fillcolor='#D6EAF8', color=COLORS['input'], penwidth='2')
    
    # 中间节点
    with dot.subgraph(name='cluster_nodes') as nodes:
        nodes.attr(label='Intermediate Nodes', style='rounded,dashed', color='#BDC3C7')
        for i in range(config.NUM_NODES):
            nodes.node(f'node_{i}', f'Node {i}\n(sum of inputs)',
                      shape='ellipse', style='filled',
                      fillcolor='#FEF3E2', color=COLORS['node'], penwidth='2')
    
    # 输出节点
    with dot.subgraph(name='cluster_output') as output:
        output.attr(label='Output', style='rounded,dashed', color='#BDC3C7')
        output.node('c_k', 'c_k\n(concat all nodes)',
                   shape='box', style='rounded,filled',
                   fillcolor=bg_color, color=color, penwidth='2')
    
    # 操作边
    op_colors = {
        'skip_connect': '#2ECC71',
        'sep_conv_3x3': '#3498DB',
        'sep_conv_5x5': '#2980B9',
        'dil_conv_3x3': '#E74C3C',
        'dil_conv_5x5': '#C0392B',
        'max_pool_3x3': '#9B59B6',
        'avg_pool_3x3': '#8E44AD',
    }
    
    for node_idx, node_edges in enumerate(cell_encoding.edges):
        for edge_idx, edge in enumerate(node_edges):
            op_name = config.OPERATIONS[edge.op_id]
            
            if op_name == 'zero':
                continue
            
            if edge.source == 0:
                source = 'c_k_2'
            elif edge.source == 1:
                source = 'c_k_1'
            else:
                source = f'node_{edge.source - 2}'
            
            target = f'node_{node_idx}'
            op_display = OP_DISPLAY_NAMES.get(op_name, op_name)
            op_color = op_colors.get(op_name, '#7F8C8D')
            
            dot.edge(source, target, label=op_display,
                    color=op_color, fontcolor=op_color, penwidth='2')
    
    # 节点到输出的连接
    for i in range(config.NUM_NODES):
        dot.edge(f'node_{i}', 'c_k', style='dashed', 
                color='#BDC3C7', penwidth='1')
    
    return dot


def create_hierarchical_diagram(normal_cell: CellEncoding, reduction_cell: CellEncoding,
                                cells_per_stage: int, num_stages: int,
                                output_format: str = 'png', dpi: int = 300,
                                show_details: bool = False):
    """
    创建层次结构图
    
    展示三个层次：
    1. 网络层（整体结构）
    2. Cell 层（Cell 类型）
    3. Node 层（节点和操作）
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装 graphviz 包。请运行: pip install graphviz")
    
    dot = Digraph(
        name='DARTS_Hierarchical',
        format=output_format,
        graph_attr={
            'rankdir': 'TB',
            'compound': 'true',
            'nodesep': '0.3',
            'ranksep': '0.8',
            'fontname': 'Arial',
            'fontsize': '12',
            'label': 'DARTS Network Hierarchical Structure\n(Network → Cells → Nodes)',
            'labelloc': 't',
            'labeljust': 'c',
            'bgcolor': 'white',
            'dpi': str(dpi),
        },
    )
    
    # ============ Layer 1: Network Level ============
    with dot.subgraph(name='cluster_network') as network:
        network.attr(
            label='Network Level',
            style='rounded,bold',
            color='#2C3E50',
            bgcolor='#F8F9FA',
            fontsize='14',
            fontcolor='#2C3E50',
        )
        
        network.node('net_input', 'Input\n(3×32×32)',
                    shape='box', style='rounded,filled',
                    fillcolor='#E8F4FD', color=COLORS['input'], penwidth='2')
        network.node('net_stem', 'Stem',
                    shape='box', style='rounded,filled',
                    fillcolor='#D6EAF8', color=COLORS['stem'], penwidth='2')
        network.node('net_cells', f'Cells\n({num_stages * cells_per_stage + num_stages - 1} total)',
                    shape='box', style='rounded,filled',
                    fillcolor='#FCF3CF', color='#F4D03F', penwidth='2')
        network.node('net_classifier', 'Classifier',
                    shape='box', style='rounded,filled',
                    fillcolor='#E8DAEF', color=COLORS['classifier'], penwidth='2')
        network.node('net_output', 'Output',
                    shape='box', style='rounded,filled',
                    fillcolor='#D5F5E3', color=COLORS['output'], penwidth='2')
        
        network.edge('net_input', 'net_stem', penwidth='1.5')
        network.edge('net_stem', 'net_cells', penwidth='1.5')
        network.edge('net_cells', 'net_classifier', penwidth='1.5')
        network.edge('net_classifier', 'net_output', penwidth='1.5')
    
    # ============ Layer 2: Cell Level ============
    with dot.subgraph(name='cluster_cells') as cells:
        cells.attr(
            label='Cell Level (Cells → composed of)',
            style='rounded,bold',
            color='#2C3E50',
            bgcolor='#F8F9FA',
            fontsize='14',
            fontcolor='#2C3E50',
        )
        
        # Normal Cell 模板
        with cells.subgraph(name='cluster_normal_template') as normal:
            normal.attr(
                label='Normal Cell Template',
                style='rounded',
                color=COLORS['normal_cell'],
                bgcolor='#D5F5E3',
            )
            normal.node('normal_c_k_2', 'c_{k-2}',
                       shape='box', style='rounded,filled',
                       fillcolor='#D6EAF8', color=COLORS['input'], penwidth='1.5',
                       width='0.8', height='0.4')
            normal.node('normal_c_k_1', 'c_{k-1}',
                       shape='box', style='rounded,filled',
                       fillcolor='#D6EAF8', color=COLORS['input'], penwidth='1.5',
                       width='0.8', height='0.4')
            
            for i in range(config.NUM_NODES):
                normal.node(f'normal_node_{i}', f'N{i}',
                           shape='circle', style='filled',
                           fillcolor='#FEF3E2', color=COLORS['node'], penwidth='1.5',
                           width='0.4', height='0.4')
            
            normal.node('normal_output', 'c_k',
                       shape='box', style='rounded,filled',
                       fillcolor='#D5F5E3', color=COLORS['normal_cell'], penwidth='1.5',
                       width='0.8', height='0.4')
            
            # 添加简化的边
            for node_idx, node_edges in enumerate(normal_cell.edges):
                for edge in node_edges:
                    op_name = config.OPERATIONS[edge.op_id]
                    if op_name == 'zero':
                        continue
                    if edge.source == 0:
                        src = 'normal_c_k_2'
                    elif edge.source == 1:
                        src = 'normal_c_k_1'
                    else:
                        src = f'normal_node_{edge.source - 2}'
                    normal.edge(src, f'normal_node_{node_idx}', 
                               color=COLORS['node'], penwidth='1')
            
            for i in range(config.NUM_NODES):
                normal.edge(f'normal_node_{i}', 'normal_output',
                           style='dashed', color='#BDC3C7', penwidth='0.5')
        
        # Reduction Cell 模板
        with cells.subgraph(name='cluster_reduction_template') as reduction:
            reduction.attr(
                label='Reduction Cell Template (stride=2)',
                style='rounded',
                color=COLORS['reduction_cell'],
                bgcolor='#FADBD8',
            )
            reduction.node('reduction_c_k_2', 'c_{k-2}',
                          shape='box', style='rounded,filled',
                          fillcolor='#D6EAF8', color=COLORS['input'], penwidth='1.5',
                          width='0.8', height='0.4')
            reduction.node('reduction_c_k_1', 'c_{k-1}',
                          shape='box', style='rounded,filled',
                          fillcolor='#D6EAF8', color=COLORS['input'], penwidth='1.5',
                          width='0.8', height='0.4')
            
            for i in range(config.NUM_NODES):
                reduction.node(f'reduction_node_{i}', f'N{i}',
                              shape='circle', style='filled',
                              fillcolor='#FEF3E2', color=COLORS['node'], penwidth='1.5',
                              width='0.4', height='0.4')
            
            reduction.node('reduction_output', 'c_k',
                          shape='box', style='rounded,filled',
                          fillcolor='#FADBD8', color=COLORS['reduction_cell'], penwidth='1.5',
                          width='0.8', height='0.4')
            
            # 添加简化的边
            for node_idx, node_edges in enumerate(reduction_cell.edges):
                for edge in node_edges:
                    op_name = config.OPERATIONS[edge.op_id]
                    if op_name == 'zero':
                        continue
                    if edge.source == 0:
                        src = 'reduction_c_k_2'
                    elif edge.source == 1:
                        src = 'reduction_c_k_1'
                    else:
                        src = f'reduction_node_{edge.source - 2}'
                    reduction.edge(src, f'reduction_node_{node_idx}',
                                  color=COLORS['node'], penwidth='1')
            
            for i in range(config.NUM_NODES):
                reduction.edge(f'reduction_node_{i}', 'reduction_output',
                              style='dashed', color='#BDC3C7', penwidth='0.5')
    
    # ============ Layer 3: Node Level ============
    with dot.subgraph(name='cluster_nodes') as nodes:
        nodes.attr(
            label='Node Level (Operations)',
            style='rounded,bold',
            color='#2C3E50',
            bgcolor='#F8F9FA',
            fontsize='14',
            fontcolor='#2C3E50',
        )
        
        # 操作类型
        ops = [
            ('skip', 'Skip Connect\n(Identity)', '#2ECC71'),
            ('sep_3', 'Sep Conv 3×3\n(Separable)', '#3498DB'),
            ('sep_5', 'Sep Conv 5×5\n(Separable)', '#2980B9'),
            ('dil_3', 'Dil Conv 3×3\n(Dilated)', '#E74C3C'),
            ('dil_5', 'Dil Conv 5×5\n(Dilated)', '#C0392B'),
            ('max_p', 'Max Pool 3×3', '#9B59B6'),
            ('avg_p', 'Avg Pool 3×3', '#8E44AD'),
            ('zero', 'Zero\n(No connect)', '#CCCCCC'),
        ]
        
        for i, (op_id, op_label, op_color) in enumerate(ops):
            nodes.node(f'op_{op_id}', op_label,
                      shape='box', style='rounded,filled',
                      fillcolor='#FFFFFF', color=op_color, penwidth='2',
                      width='1.2')
    
    # 添加层次间的虚线连接
    dot.edge('net_cells', 'normal_output', 
            ltail='cluster_network', lhead='cluster_cells',
            style='dashed', color='#BDC3C7', penwidth='1',
            label='composed of')
    
    dot.edge('normal_node_0', 'op_sep_3',
            ltail='cluster_normal_template', lhead='cluster_nodes',
            style='dashed', color='#BDC3C7', penwidth='1',
            label='uses operations')
    
    return dot


def create_compact_overview(cells_per_stage: int, num_stages: int,
                           output_format: str = 'png', dpi: int = 300):
    """
    创建紧凑的网络概览图
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装 graphviz 包。请运行: pip install graphviz")
    
    total_cells = num_stages * cells_per_stage + (num_stages - 1)
    
    dot = Digraph(
        name='DARTS_Compact',
        format=output_format,
        graph_attr={
            'rankdir': 'LR',
            'nodesep': '0.3',
            'ranksep': '0.5',
            'fontname': 'Arial',
            'fontsize': '12',
            'label': f'DARTS Network ({total_cells} Cells)',
            'labelloc': 't',
            'bgcolor': 'white',
            'dpi': str(dpi),
        },
    )
    
    # 简化的流程图
    dot.node('in', 'IN', shape='circle', style='filled', 
            fillcolor=COLORS['input'], fontcolor='white', width='0.5')
    dot.node('stem', 'Stem', shape='box', style='rounded,filled',
            fillcolor=COLORS['stem'], fontcolor='white')
    
    prev = 'stem'
    dot.edge('in', 'stem', penwidth='2')
    
    for stage in range(num_stages):
        if stage > 0:
            # Reduction Cell
            r_name = f'R{stage}'
            dot.node(r_name, f'R', shape='box', style='rounded,filled',
                    fillcolor=COLORS['reduction_cell'], fontcolor='white',
                    width='0.4')
            dot.edge(prev, r_name, penwidth='2')
            prev = r_name
        
        # Normal Cells (grouped)
        n_name = f'N_stage{stage}'
        dot.node(n_name, f'N×{cells_per_stage}', shape='box', style='rounded,filled',
                fillcolor=COLORS['normal_cell'], fontcolor='white')
        dot.edge(prev, n_name, penwidth='2')
        prev = n_name
    
    dot.node('gap', 'GAP', shape='box', style='rounded,filled',
            fillcolor=COLORS['classifier'], fontcolor='white', width='0.5')
    dot.node('fc', 'FC', shape='box', style='rounded,filled',
            fillcolor=COLORS['classifier'], fontcolor='white', width='0.4')
    dot.node('out', 'OUT', shape='circle', style='filled',
            fillcolor=COLORS['output'], fontcolor='white', width='0.5')
    
    dot.edge(prev, 'gap', penwidth='2')
    dot.edge('gap', 'fc', penwidth='2')
    dot.edge('fc', 'out', penwidth='2')
    
    # 图例
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='#BDC3C7')
        legend.node('leg_n', 'N = Normal Cell', shape='plaintext')
        legend.node('leg_r', 'R = Reduction Cell', shape='plaintext')
        legend.node('leg_gap', 'GAP = Global Avg Pool', shape='plaintext')
        legend.node('leg_fc', 'FC = Fully Connected', shape='plaintext')
    
    return dot


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print(" DARTS 网络架构层次图可视化")
    print("=" * 60)
    
    # 1. 加载编码
    print("\n[1/4] 加载 Cell 编码...")
    
    if args.encoding_file:
        print(f"  从文件加载: {args.encoding_file}")
        normal_encoding, reduction_encoding = load_encoding_from_file(args.encoding_file)
    else:
        print("  从命令行参数加载")
        normal_encoding = parse_cell_encoding(args.normal_cell)
        reduction_encoding = parse_cell_encoding(args.reduction_cell)
    
    # 2. 创建 CellEncoding 对象
    try:
        normal_cell = CellEncoding.from_list(normal_encoding)
        reduction_cell = CellEncoding.from_list(reduction_encoding)
    except ValueError as e:
        print(f"错误: 编码格式不正确 - {e}")
        sys.exit(1)
    
    # 获取网络参数
    cells_per_stage = args.cells_per_stage or config.CELLS_PER_STAGE
    num_stages = args.num_stages or config.NUM_STAGES
    total_cells = num_stages * cells_per_stage + (num_stages - 1)
    
    print(f"  Normal Cell 编码: {normal_encoding}")
    print(f"  Reduction Cell 编码: {reduction_encoding}")
    print(f"  网络结构: {num_stages} stages × {cells_per_stage} cells + {num_stages - 1} reduction = {total_cells} cells")
    
    # 3. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 生成可视化
    print("\n[2/4] 生成网络概览图...")
    output_files = []
    
    # 网络概览图
    if args.compact:
        dot = create_compact_overview(cells_per_stage, num_stages, args.format, args.dpi)
        output_path = os.path.join(args.output_dir, f'{args.output_name}_compact')
    else:
        dot = create_network_overview(cells_per_stage, num_stages, args.format, args.dpi)
        output_path = os.path.join(args.output_dir, f'{args.output_name}_overview')
    
    rendered = dot.render(output_path, cleanup=True)
    output_files.append(rendered)
    print(f"  已保存: {rendered}")
    
    # 层次结构图
    print("\n[3/4] 生成层次结构图...")
    dot = create_hierarchical_diagram(
        normal_cell, reduction_cell, cells_per_stage, num_stages,
        args.format, args.dpi, args.show_details
    )
    output_path = os.path.join(args.output_dir, f'{args.output_name}_hierarchical')
    rendered = dot.render(output_path, cleanup=True)
    output_files.append(rendered)
    print(f"  已保存: {rendered}")
    
    # Cell 详细图 (如果需要)
    if args.show_details:
        print("\n[3.5/4] 生成 Cell 详细结构图...")
        
        dot = create_cell_detail_diagram(normal_cell, 'Normal', args.format, args.dpi)
        output_path = os.path.join(args.output_dir, f'{args.output_name}_normal_detail')
        rendered = dot.render(output_path, cleanup=True)
        output_files.append(rendered)
        print(f"  已保存: {rendered}")
        
        dot = create_cell_detail_diagram(reduction_cell, 'Reduction', args.format, args.dpi)
        output_path = os.path.join(args.output_dir, f'{args.output_name}_reduction_detail')
        rendered = dot.render(output_path, cleanup=True)
        output_files.append(rendered)
        print(f"  已保存: {rendered}")
    
    # 5. 完成
    print("\n[4/4] 完成!")
    print(f"\n生成的文件:")
    for f in output_files:
        print(f"  - {f}")
    
    print("\n" + "=" * 60)
    print(" 可视化完成!")
    print("=" * 60)
    
    return output_files


if __name__ == '__main__':
    main()
