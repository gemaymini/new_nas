# -*- coding: utf-8 -*-
"""
DARTS 网络构建模块
实现 DARTS 操作、Cell 和网络堆叠
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from configuration.config import config
from core.encoding import CellEncoding, Individual
from utils.logger import logger


# ==================== DARTS 操作定义 ====================

class ReLUConvBN(nn.Module):
    """ReLU-Conv-BN 序列"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """深度可分离卷积: ReLU-SepConv-BN-ReLU-SepConv-BN"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """空洞可分离卷积: ReLU-DilSepConv-BN"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, dilation: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """恒等映射 (Skip Connection)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class Zero(nn.Module):
    """零操作: 返回零张量"""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        # 如果 stride=2，需要下采样
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """分解式下采样: 用于 Reduction Cell 的输入处理"""
    def __init__(self, C_in: int, C_out: int, affine: bool = True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x):
        x = self.relu(x)
        # 边界检查：当特征图尺寸 <= 1 时，使用常规方法
        if x.size(2) <= 1 or x.size(3) <= 1:
            out = torch.cat([self.conv_1(x), self.conv_1(x)], dim=1)
        else:
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def get_op(op_name: str, C: int, stride: int, affine: bool = True) -> nn.Module:
    """
    根据操作名称获取操作模块
    
    Args:
        op_name: 操作名称
        C: 通道数 (输入=输出)
        stride: 步长 (Normal Cell 为 1, Reduction Cell 为 2)
        affine: BatchNorm 是否使用 affine
    
    Returns:
        操作模块
    """
    OPS = {
        'zero': lambda C, stride, affine: Zero(stride),
        'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
        'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
        'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
        'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, dilation=2, affine=affine),
        'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, dilation=2, affine=affine),
        'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
        'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    }
    
    if op_name not in OPS:
        raise ValueError(f"Unknown operation: {op_name}")
    
    return OPS[op_name](C, stride, affine)


# ==================== Cell 定义 ====================

class Cell(nn.Module):
    """
    DARTS Cell 实现
    
    一个 Cell 是一个 DAG:
    - 2 个输入节点 (s0, s1)
    - N 个中间节点
    - 输出为所有中间节点的 concatenation
    """
    
    def __init__(self, cell_encoding: CellEncoding, C_prev_prev: int, C_prev: int, C: int, reduction: bool, reduction_prev: bool):
        """
        Args:
            cell_encoding: Cell 的 DAG 编码
            C_prev_prev: c_{k-2} 的通道数
            C_prev: c_{k-1} 的通道数
            C: 当前 Cell 的输出通道数
            reduction: 是否是 Reduction Cell
            reduction_prev: 前一个 Cell 是否是 Reduction Cell
        """
        super().__init__()
        
        self.reduction = reduction
        self.num_nodes = config.NUM_NODES
        
        # 预处理层: 将输入统一到相同的通道数和空间尺寸
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        # 构建操作
        self._ops = nn.ModuleList()
        self._edge_indices = []  # 记录每个操作对应的 (node_idx, input_idx, source)
        
        stride = 2 if reduction else 1
        
        for node_idx, node_edges in enumerate(cell_encoding.edges):
            for edge_idx, edge in enumerate(node_edges):
                op_name = config.OPERATIONS[edge.op_id]
                # 只有连接到输入节点且是 reduction cell 时才使用 stride=2
                if edge.source < 2 and reduction:
                    op_stride = stride
                else:
                    op_stride = 1
                op = get_op(op_name, C, op_stride)
                self._ops.append(op)
                self._edge_indices.append((node_idx, edge_idx, edge.source))
        
        # 计算输出通道数 (所有中间节点的 concat)
        self.out_channels = self.num_nodes * C
        
        # 将编码保存用于调试
        self.cell_encoding = cell_encoding
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            s0: c_{k-2} 的输出
            s1: c_{k-1} 的输出
        
        Returns:
            Cell 输出 (所有中间节点的 concat)
        """
        # 预处理
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        # 状态列表: [Input_0, Input_1, Node_0, Node_1, ...]
        states = [s0, s1]
        
        # 逐节点计算
        op_idx = 0
        for node_idx in range(self.num_nodes):
            # 收集该节点的所有输入
            node_inputs = []
            edges_per_node = config.EDGES_PER_NODE
            
            for edge_idx in range(edges_per_node):
                _, _, source = self._edge_indices[op_idx]
                h = states[source]
                h = self._ops[op_idx](h)
                node_inputs.append(h)
                op_idx += 1
            
            # 该节点的输出是所有输入的和
            node_output = sum(node_inputs)
            states.append(node_output)
        
        # 输出: 所有中间节点的 concat
        return torch.cat(states[2:], dim=1)


# ==================== Network 定义 ====================

class DARTSNetwork(nn.Module):
    """
    DARTS 网络实现
    
    结构:
    - Stem: 初始卷积层
    - Cells: Normal Cell 和 Reduction Cell 交替堆叠
    - Classifier: Global Pooling + FC
    """
    
    def __init__(self, normal_cell: CellEncoding, reduction_cell: CellEncoding, 
                 num_classes: int = 10, init_channels: int = None,
                 cells_per_stage: int = None, num_stages: int = None,
                 enable_dropout: bool = True):
        """
        Args:
            normal_cell: Normal Cell 编码
            reduction_cell: Reduction Cell 编码
            num_classes: 分类数
            init_channels: 初始通道数
            cells_per_stage: 每个 Stage 的 Cell 数量
            num_stages: Stage 数量
            enable_dropout: 是否启用 Dropout (NTK 评估时应设为 False)
        """
        super().__init__()
        
        self.enable_dropout = enable_dropout
        
        self.num_classes = num_classes
        C = init_channels or config.INIT_CHANNELS
        cells_per_stage = cells_per_stage or config.CELLS_PER_STAGE
        num_stages = num_stages or config.NUM_STAGES
        
        # 计算 Cell 总数
        self.num_cells = num_stages * cells_per_stage + (num_stages - 1)  # reduction cells
        
        # Stem: 3 -> C*3
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 3)
        )
        
        # 构建 Cells
        self.cells = nn.ModuleList()
        
        C_prev_prev = C * 3
        C_prev = C * 3
        C_curr = C
        reduction_prev = False
        
        cell_idx = 0
        for stage_idx in range(num_stages):
            # 每个 Stage 开始时添加 Reduction Cell (除了第一个 Stage)
            if stage_idx > 0:
                # Reduction Cell
                reduction = True
                cell = Cell(reduction_cell, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                self.cells.append(cell)
                
                C_prev_prev = C_prev
                C_prev = cell.out_channels
                C_curr *= 2  # 通道数翻倍
                reduction_prev = True
                cell_idx += 1
            
            # Normal Cells
            for _ in range(cells_per_stage):
                reduction = False
                cell = Cell(normal_cell, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                self.cells.append(cell)
                
                C_prev_prev = C_prev
                C_prev = cell.out_channels
                reduction_prev = False
                cell_idx += 1
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)  # Dropout 正则化
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # 保存编码
        self.normal_cell_encoding = normal_cell
        self.reduction_cell_encoding = reduction_cell
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        s0 = s1 = self.stem(x)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        if self.enable_dropout:
            out = self.dropout(out)  # 仅在训练时应用 Dropout
        out = self.classifier(out)
        
        return out
    
    def get_param_count(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())


# ==================== Network Builder ====================

class NetworkBuilder:
    """网络构建器"""
    
    @staticmethod
    def build(normal_cell: CellEncoding, reduction_cell: CellEncoding, 
              num_classes: int = 10, **kwargs) -> DARTSNetwork:
        """
        从 Cell 编码构建网络
        
        Args:
            normal_cell: Normal Cell 编码
            reduction_cell: Reduction Cell 编码
            num_classes: 分类数
            **kwargs: 其他参数 (init_channels, cells_per_stage, num_stages)
        
        Returns:
            DARTSNetwork 实例
        """
        return DARTSNetwork(normal_cell, reduction_cell, num_classes, **kwargs)
    
    @staticmethod
    def build_from_individual(individual: Individual, num_classes: int = 10, **kwargs) -> DARTSNetwork:
        """
        从个体构建网络
        
        Args:
            individual: 个体
            num_classes: 分类数
            **kwargs: 其他参数
        
        Returns:
            DARTSNetwork 实例
        """
        return NetworkBuilder.build(
            individual.normal_cell, 
            individual.reduction_cell, 
            num_classes, 
            **kwargs
        )
    
    @staticmethod
    def build_from_encoding(encoding: dict, num_classes: int = 10, **kwargs) -> DARTSNetwork:
        """
        从编码字典构建网络 (兼容旧接口)
        
        Args:
            encoding: {'normal_cell': [...], 'reduction_cell': [...]}
            num_classes: 分类数
            **kwargs: 其他参数
        
        Returns:
            DARTSNetwork 实例
        """
        normal_cell = CellEncoding.from_list(encoding['normal_cell'])
        reduction_cell = CellEncoding.from_list(encoding['reduction_cell'])
        return NetworkBuilder.build(normal_cell, reduction_cell, num_classes, **kwargs)
    
    @staticmethod
    def calculate_param_count(individual: Individual, num_classes: int = 10) -> int:
        """计算参数量"""
        network = NetworkBuilder.build_from_individual(individual, num_classes)
        return network.get_param_count()
    
    @staticmethod
    def test_forward(individual: Individual, input_size: tuple = None) -> bool:
        """测试前向传播"""
        if input_size is None:
            input_size = config.NTK_INPUT_SIZE
        
        try:
            network = NetworkBuilder.build_from_individual(individual)
            network.eval()
            x = torch.randn(1, *input_size)
            with torch.no_grad():
                output = network(x)
            
            expected_output_size = (1, config.NTK_NUM_CLASSES)
            return output.shape == expected_output_size
        
        except Exception as e:
            logger.error(f"Forward test failed: {e}")
            return False
