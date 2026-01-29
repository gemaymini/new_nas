# -*- coding: utf-8 -*-
"""
DARTS Cell 结构可视化模块

根据 Cell 编码绘制 DAG 结构图，支持输出为 PNG、PDF、SVG 等格式。

使用方法:
1. 从命令行指定编码:
   python visualize_cell.py --normal_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3" --reduction_cell "0,2,1,3,0,4,2,5,1,6,3,7,2,2,0,3"

2. 从 JSON 文件读取:
   python visualize_cell.py --encoding_file path/to/encoding.json

3. 仅绘制单个 Cell:
   python visualize_cell.py --encoding_file path/to/encoding.json --cell_type normal

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

# 操作对应的边颜色 (用于区分不同操作类型)
OP_COLORS = {
    'zero': '#CCCCCC',        # 灰色 - 无连接
    'skip_connect': '#2ECC71', # 绿色 - 跳跃连接
    'sep_conv_3x3': '#3498DB', # 蓝色 - 可分离卷积
    'sep_conv_5x5': '#2980B9', # 深蓝色
    'dil_conv_3x3': '#E74C3C', # 红色 - 空洞卷积
    'dil_conv_5x5': '#C0392B', # 深红色
    'max_pool_3x3': '#9B59B6', # 紫色 - 池化
    'avg_pool_3x3': '#8E44AD', # 深紫色
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='可视化 DARTS Cell 结构',
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
    
    # 可视化选项
    parser.add_argument(
        '--cell_type', type=str, default='both',
        choices=['normal', 'reduction', 'both'],
        help='要绘制的 Cell 类型 (默认: both)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./visualizations',
        help='输出目录 (默认: ./visualizations)'
    )
    parser.add_argument(
        '--output_name', type=str, default=None,
        help='输出文件名前缀 (默认: cell_structure)'
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
        '--show_zero', action='store_true',
        help='显示 zero 操作的边 (默认隐藏)'
    )
    parser.add_argument(
        '--horizontal', action='store_true',
        help='水平布局 (默认: 垂直布局)'
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


def create_cell_graph(cell_encoding: CellEncoding, cell_name: str, 
                      show_zero: bool = False, horizontal: bool = False):
    """
    创建 Cell 的 Graphviz 图
    
    Args:
        cell_encoding: Cell 编码对象
        cell_name: Cell 名称 (用于标题)
        show_zero: 是否显示 zero 操作
        horizontal: 是否水平布局
    
    Returns:
        graphviz.Digraph 对象
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "需要安装 graphviz 包。请运行: pip install graphviz\n"
            "同时需要安装 Graphviz 系统软件: https://graphviz.org/download/"
        )
    
    # 创建有向图
    dot = Digraph(
        name=cell_name,
        format='png',
        graph_attr={
            'rankdir': 'LR' if horizontal else 'TB',
            'splines': 'ortho',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'fontname': 'Arial',
            'fontsize': '14',
            'label': f'{cell_name}',
            'labelloc': 't',
            'labeljust': 'c',
            'bgcolor': 'white',
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
    
    # 添加输入节点
    with dot.subgraph(name='cluster_inputs') as inputs:
        inputs.attr(style='invis')
        inputs.node('input_0', 'c_{k-2}', 
                   shape='box', style='rounded,filled', 
                   fillcolor='#E8F4FD', color='#3498DB', penwidth='2')
        inputs.node('input_1', 'c_{k-1}', 
                   shape='box', style='rounded,filled', 
                   fillcolor='#E8F4FD', color='#3498DB', penwidth='2')
    
    # 添加中间节点
    for i in range(config.NUM_NODES):
        dot.node(f'node_{i}', f'Node {i}',
                shape='ellipse', style='filled',
                fillcolor='#FEF3E2', color='#E67E22', penwidth='2')
    
    # 添加输出节点
    dot.node('output', 'concat',
            shape='box', style='rounded,filled',
            fillcolor='#E8F8F5', color='#27AE60', penwidth='2')
    
    # 添加边 (根据编码)
    for node_idx, node_edges in enumerate(cell_encoding.edges):
        for edge_idx, edge in enumerate(node_edges):
            op_name = config.OPERATIONS[edge.op_id]
            
            # 跳过 zero 操作 (除非指定显示)
            if op_name == 'zero' and not show_zero:
                continue
            
            # 确定源节点名称
            if edge.source == 0:
                source_name = 'input_0'
            elif edge.source == 1:
                source_name = 'input_1'
            else:
                source_name = f'node_{edge.source - 2}'
            
            target_name = f'node_{node_idx}'
            
            # 获取操作显示名和颜色
            op_display = OP_DISPLAY_NAMES.get(op_name, op_name)
            op_color = OP_COLORS.get(op_name, '#333333')
            
            # 设置边样式
            edge_style = 'dashed' if op_name == 'zero' else 'solid'
            edge_width = '1' if op_name == 'zero' else '2'
            
            dot.edge(source_name, target_name, 
                    label=op_display,
                    color=op_color,
                    fontcolor=op_color,
                    style=edge_style,
                    penwidth=edge_width)
    
    # 添加从中间节点到输出的边
    for i in range(config.NUM_NODES):
        dot.edge(f'node_{i}', 'output', 
                style='dashed', color='#95A5A6', penwidth='1')
    
    return dot


def create_combined_graph(normal_cell: CellEncoding, reduction_cell: CellEncoding,
                          show_zero: bool = False, horizontal: bool = False):
    """
    创建包含 Normal 和 Reduction Cell 的组合图
    
    Args:
        normal_cell: Normal Cell 编码
        reduction_cell: Reduction Cell 编码
        show_zero: 是否显示 zero 操作
        horizontal: 是否水平布局
    
    Returns:
        graphviz.Digraph 对象
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "需要安装 graphviz 包。请运行: pip install graphviz\n"
            "同时需要安装 Graphviz 系统软件: https://graphviz.org/download/"
        )
    
    # 创建主图
    dot = Digraph(
        name='DARTS_Cells',
        format='png',
        graph_attr={
            'rankdir': 'LR' if horizontal else 'TB',
            'splines': 'ortho',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'fontname': 'Arial',
            'fontsize': '16',
            'label': 'DARTS Cell Architecture',
            'labelloc': 't',
            'compound': 'true',
        }
    )
    
    # 创建 Normal Cell 子图
    with dot.subgraph(name='cluster_normal') as normal:
        normal.attr(
            label='Normal Cell',
            style='rounded',
            color='#3498DB',
            bgcolor='#F8FBFD',
            fontcolor='#2C3E50',
            penwidth='2'
        )
        _add_cell_nodes(normal, normal_cell, 'normal', show_zero)
    
    # 创建 Reduction Cell 子图
    with dot.subgraph(name='cluster_reduction') as reduction:
        reduction.attr(
            label='Reduction Cell',
            style='rounded',
            color='#E74C3C',
            bgcolor='#FDF8F8',
            fontcolor='#2C3E50',
            penwidth='2'
        )
        _add_cell_nodes(reduction, reduction_cell, 'reduction', show_zero)
    
    return dot


def _add_cell_nodes(graph, cell_encoding: CellEncoding, prefix: str, show_zero: bool):
    """向子图添加节点和边"""
    
    # 添加输入节点
    graph.node(f'{prefix}_input_0', 'c_{k-2}', 
               shape='box', style='rounded,filled', 
               fillcolor='#E8F4FD', color='#3498DB', penwidth='2',
               fontname='Arial', fontsize='10')
    graph.node(f'{prefix}_input_1', 'c_{k-1}', 
               shape='box', style='rounded,filled', 
               fillcolor='#E8F4FD', color='#3498DB', penwidth='2',
               fontname='Arial', fontsize='10')
    
    # 添加中间节点
    for i in range(config.NUM_NODES):
        graph.node(f'{prefix}_node_{i}', f'{i}',
                  shape='circle', style='filled',
                  fillcolor='#FEF3E2', color='#E67E22', penwidth='2',
                  width='0.4', height='0.4',
                  fontname='Arial', fontsize='10')
    
    # 添加输出节点
    graph.node(f'{prefix}_output', 'c_{k}',
              shape='box', style='rounded,filled',
              fillcolor='#E8F8F5', color='#27AE60', penwidth='2',
              fontname='Arial', fontsize='10')
    
    # 添加边
    for node_idx, node_edges in enumerate(cell_encoding.edges):
        for edge_idx, edge in enumerate(node_edges):
            op_name = config.OPERATIONS[edge.op_id]
            
            if op_name == 'zero' and not show_zero:
                continue
            
            if edge.source == 0:
                source_name = f'{prefix}_input_0'
            elif edge.source == 1:
                source_name = f'{prefix}_input_1'
            else:
                source_name = f'{prefix}_node_{edge.source - 2}'
            
            target_name = f'{prefix}_node_{node_idx}'
            
            op_display = OP_DISPLAY_NAMES.get(op_name, op_name)
            op_color = OP_COLORS.get(op_name, '#333333')
            edge_style = 'dashed' if op_name == 'zero' else 'solid'
            edge_width = '1' if op_name == 'zero' else '1.5'
            
            graph.edge(source_name, target_name, 
                      label=op_display,
                      color=op_color,
                      fontcolor=op_color,
                      style=edge_style,
                      penwidth=edge_width,
                      fontname='Arial',
                      fontsize='8')
    
    # 从中间节点到输出的边
    for i in range(config.NUM_NODES):
        graph.edge(f'{prefix}_node_{i}', f'{prefix}_output', 
                  style='dashed', color='#BDC3C7', penwidth='1')


def visualize_cell_matplotlib(cell_encoding: CellEncoding, cell_name: str,
                               output_path: str, show_zero: bool = False,
                               figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
    """
    使用 Matplotlib 可视化 Cell (备选方案，不需要 Graphviz)
    
    Args:
        cell_encoding: Cell 编码对象
        cell_name: Cell 名称
        output_path: 输出文件路径
        show_zero: 是否显示 zero 操作
        figsize: 图像大小
        dpi: 图像 DPI
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(cell_name, fontsize=16, fontweight='bold', pad=20)
    
    # 节点位置
    node_positions = {
        'input_0': (0.5, 6),
        'input_1': (0.5, 4),
    }
    
    # 中间节点位置 (根据节点数动态计算)
    num_nodes = config.NUM_NODES
    for i in range(num_nodes):
        x = 3 + (i % 2) * 2
        y = 6 - (i // 2) * 2.5
        node_positions[f'node_{i}'] = (x, y)
    
    node_positions['output'] = (8, 4)
    
    # 绘制节点
    node_colors = {
        'input': '#E8F4FD',
        'intermediate': '#FEF3E2',
        'output': '#E8F8F5'
    }
    
    # 输入节点
    for i, name in enumerate(['input_0', 'input_1']):
        x, y = node_positions[name]
        bbox = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor=node_colors['input'],
                              edgecolor='#3498DB', linewidth=2)
        ax.add_patch(bbox)
        label = f'c_{{k-{2-i}}}'
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    
    # 中间节点
    for i in range(num_nodes):
        x, y = node_positions[f'node_{i}']
        circle = plt.Circle((x, y), 0.35, 
                            facecolor=node_colors['intermediate'],
                            edgecolor='#E67E22', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 输出节点
    x, y = node_positions['output']
    bbox = FancyBboxPatch((x-0.5, y-0.3), 1.0, 0.6,
                          boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=node_colors['output'],
                          edgecolor='#27AE60', linewidth=2)
    ax.add_patch(bbox)
    ax.text(x, y, 'c_k', ha='center', va='center', fontsize=10)
    
    # 绘制边
    for node_idx, node_edges in enumerate(cell_encoding.edges):
        for edge_idx, edge in enumerate(node_edges):
            op_name = config.OPERATIONS[edge.op_id]
            
            if op_name == 'zero' and not show_zero:
                continue
            
            # 确定源和目标位置
            if edge.source == 0:
                source_pos = node_positions['input_0']
            elif edge.source == 1:
                source_pos = node_positions['input_1']
            else:
                source_pos = node_positions[f'node_{edge.source - 2}']
            
            target_pos = node_positions[f'node_{node_idx}']
            
            # 获取颜色和样式
            op_color = OP_COLORS.get(op_name, '#333333')
            linestyle = '--' if op_name == 'zero' else '-'
            
            # 绘制箭头
            ax.annotate('', xy=target_pos, xytext=source_pos,
                       arrowprops=dict(arrowstyle='->', color=op_color,
                                      linestyle=linestyle, linewidth=1.5,
                                      shrinkA=15, shrinkB=15))
            
            # 添加操作标签
            mid_x = (source_pos[0] + target_pos[0]) / 2
            mid_y = (source_pos[1] + target_pos[1]) / 2
            op_display = OP_DISPLAY_NAMES.get(op_name, op_name)
            ax.text(mid_x, mid_y, op_display, ha='center', va='center',
                   fontsize=8, color=op_color, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='none', alpha=0.8))
    
    # 从中间节点到输出的边
    for i in range(num_nodes):
        source_pos = node_positions[f'node_{i}']
        target_pos = node_positions['output']
        ax.annotate('', xy=target_pos, xytext=source_pos,
                   arrowprops=dict(arrowstyle='->', color='#BDC3C7',
                                  linestyle='--', linewidth=1,
                                  shrinkA=15, shrinkB=15))
    
    # 添加图例
    legend_elements = []
    for op_name, color in OP_COLORS.items():
        if op_name == 'zero' and not show_zero:
            continue
        display_name = OP_DISPLAY_NAMES.get(op_name, op_name)
        legend_elements.append(mpatches.Patch(color=color, label=display_name))
    
    ax.legend(handles=legend_elements, loc='lower right', 
             fontsize=8, ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print(" DARTS Cell 结构可视化")
    print("=" * 60)
    
    # 1. 加载编码
    print("\n[1/3] 加载 Cell 编码...")
    
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
    
    print(f"  Normal Cell 编码: {normal_encoding}")
    print(f"  Reduction Cell 编码: {reduction_encoding}")
    
    # 3. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or 'cell_structure'
    
    # 4. 生成可视化
    print("\n[2/3] 生成可视化图...")
    
    output_files = []
    
    try:
        # 尝试使用 Graphviz
        from graphviz import Digraph
        use_graphviz = True
        print("  使用 Graphviz 渲染...")
    except ImportError:
        use_graphviz = False
        print("  Graphviz 未安装，使用 Matplotlib 渲染...")
        print("  (提示: 安装 graphviz 可获得更好的渲染效果)")
    
    if use_graphviz:
        if args.cell_type in ['normal', 'both']:
            dot = create_cell_graph(normal_cell, 'Normal Cell', 
                                   args.show_zero, args.horizontal)
            output_path = os.path.join(args.output_dir, f'{output_name}_normal')
            dot.attr(dpi=str(args.dpi))
            dot.format = args.format
            rendered_path = dot.render(output_path, cleanup=True)
            output_files.append(rendered_path)
            print(f"  已保存: {rendered_path}")
        
        if args.cell_type in ['reduction', 'both']:
            dot = create_cell_graph(reduction_cell, 'Reduction Cell',
                                   args.show_zero, args.horizontal)
            output_path = os.path.join(args.output_dir, f'{output_name}_reduction')
            dot.attr(dpi=str(args.dpi))
            dot.format = args.format
            rendered_path = dot.render(output_path, cleanup=True)
            output_files.append(rendered_path)
            print(f"  已保存: {rendered_path}")
        
        if args.cell_type == 'both':
            # 创建组合图
            dot = create_combined_graph(normal_cell, reduction_cell,
                                       args.show_zero, args.horizontal)
            output_path = os.path.join(args.output_dir, f'{output_name}_combined')
            dot.attr(dpi=str(args.dpi))
            dot.format = args.format
            rendered_path = dot.render(output_path, cleanup=True)
            output_files.append(rendered_path)
            print(f"  已保存: {rendered_path}")
    
    else:
        # 使用 Matplotlib 作为备选
        if args.cell_type in ['normal', 'both']:
            output_path = os.path.join(args.output_dir, f'{output_name}_normal.{args.format}')
            visualize_cell_matplotlib(normal_cell, 'Normal Cell', output_path,
                                      args.show_zero, dpi=args.dpi)
            output_files.append(output_path)
            print(f"  已保存: {output_path}")
        
        if args.cell_type in ['reduction', 'both']:
            output_path = os.path.join(args.output_dir, f'{output_name}_reduction.{args.format}')
            visualize_cell_matplotlib(reduction_cell, 'Reduction Cell', output_path,
                                      args.show_zero, dpi=args.dpi)
            output_files.append(output_path)
            print(f"  已保存: {output_path}")
    
    # 5. 完成
    print("\n[3/3] 完成!")
    print(f"\n生成的文件:")
    for f in output_files:
        print(f"  - {f}")
    
    print("\n" + "=" * 60)
    print(" 可视化完成!")
    print("=" * 60)
    
    return output_files


if __name__ == '__main__':
    main()
