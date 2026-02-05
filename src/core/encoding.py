# -*- coding: utf-8 -*-
"""
DARTS Cell 编码模块
实现基于 DAG 的 Cell 编码策略
"""
import random
import copy
from typing import List, Tuple, Optional
from dataclasses import dataclass
from configuration.config import config


@dataclass
class Edge:
    """
    边定义: 来源节点索引 + 操作类型ID
    
    source: 来源节点索引
        - 0: Input_0 (c_{k-2})
        - 1: Input_1 (c_{k-1})
        - 2+: 中间节点 Node_0, Node_1, ...
    op_id: 操作 ID (0-7)
    """
    source: int
    op_id: int
    
    def to_list(self) -> List[int]:
        return [self.source, self.op_id]
    
    @classmethod
    def from_list(cls, data: List[int]) -> 'Edge':
        return cls(source=data[0], op_id=data[1])
    
    def copy(self) -> 'Edge':
        return Edge(source=self.source, op_id=self.op_id)


class CellEncoding:
    """
    Cell 编码类
    
    一个 Cell 是一个 DAG，包含:
    - 2 个输入节点 (Input_0, Input_1)
    - N 个中间节点 (Node_0, ..., Node_{N-1})
    - 1 个输出节点 (所有中间节点的 concat)
    
    每个中间节点有 K 条输入边，每条边定义:
    - 来源节点索引
    - 操作类型 ID
    
    编码格式: [source_0, op_0, source_1, op_1, ...]
    编码长度: N * K * 2
    """
    
    def __init__(self, edges: List[List[Edge]] = None):
        """
        Args:
            edges: edges[node_idx] = [edge_0, edge_1, ...], 每个节点的输入边列表
        """
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        
        if edges is not None:
            self.edges = edges
        else:
            self.edges = self._random_edges()
    
    def _random_edges(self) -> List[List[Edge]]:
        """随机生成所有边"""
        edges = []
        num_ops = len(config.OPERATIONS)
        
        for node_idx in range(self.num_nodes):
            # 可选来源: Input_0, Input_1, Node_0, ..., Node_{node_idx-1}
            valid_sources = list(range(2 + node_idx))
            node_edges = []
            
            for _ in range(self.edges_per_node):
                source = random.choice(valid_sources)
                op_id = random.randint(0, num_ops - 1)
                node_edges.append(Edge(source=source, op_id=op_id))
            
            edges.append(node_edges)
        
        return edges
    
    def to_list(self) -> List[int]:
        """转换为整数列表"""
        result = []
        for node_edges in self.edges:
            for edge in node_edges:
                result.extend(edge.to_list())
        return result
    
    @classmethod
    def from_list(cls, encoding: List[int]) -> 'CellEncoding':
        """从整数列表解码"""
        num_nodes = config.NUM_NODES
        edges_per_node = config.EDGES_PER_NODE
        expected_length = num_nodes * edges_per_node * 2
        
        if len(encoding) != expected_length:
            raise ValueError(f"Expected encoding length {expected_length}, got {len(encoding)}")
        
        edges = []
        idx = 0
        
        for node_idx in range(num_nodes):
            node_edges = []
            for _ in range(edges_per_node):
                edge_data = encoding[idx:idx+2]
                node_edges.append(Edge.from_list(edge_data))
                idx += 2
            edges.append(node_edges)
        
        cell = cls.__new__(cls)
        cell.num_nodes = num_nodes
        cell.edges_per_node = edges_per_node
        cell.edges = edges
        return cell
    
    def copy(self) -> 'CellEncoding':
        """深拷贝"""
        new_edges = []
        for node_edges in self.edges:
            new_edges.append([edge.copy() for edge in node_edges])
        
        cell = CellEncoding.__new__(CellEncoding)
        cell.num_nodes = self.num_nodes
        cell.edges_per_node = self.edges_per_node
        cell.edges = new_edges
        return cell
    
    def validate(self) -> bool:
        """验证编码有效性"""
        num_ops = len(config.OPERATIONS)
        
        for node_idx, node_edges in enumerate(self.edges):
            if len(node_edges) != self.edges_per_node:
                return False
            
            valid_sources = list(range(2 + node_idx))
            
            for edge in node_edges:
                if edge.source not in valid_sources:
                    return False
                if edge.op_id < 0 or edge.op_id >= num_ops:
                    return False
        
        return True
    
    def get_edge(self, node_idx: int, edge_idx: int) -> Edge:
        """获取指定边"""
        return self.edges[node_idx][edge_idx]
    
    def set_edge(self, node_idx: int, edge_idx: int, edge: Edge):
        """设置指定边"""
        self.edges[node_idx][edge_idx] = edge
    
    def get_valid_sources(self, node_idx: int) -> List[int]:
        """获取指定节点的有效输入来源"""
        return list(range(2 + node_idx))
    
    def __repr__(self):
        lines = [f"CellEncoding(num_nodes={self.num_nodes}, edges_per_node={self.edges_per_node}):"]
        for node_idx, node_edges in enumerate(self.edges):
            edge_strs = []
            for edge in node_edges:
                op_name = config.OPERATIONS[edge.op_id] if edge.op_id < len(config.OPERATIONS) else f"op_{edge.op_id}"
                source_name = f"Input_{edge.source}" if edge.source < 2 else f"Node_{edge.source - 2}"
                edge_strs.append(f"{source_name}→{op_name}")
            lines.append(f"  Node_{node_idx}: {', '.join(edge_strs)}")
        return "\n".join(lines)
    
    @staticmethod
    def encoding_length() -> int:
        """返回单个 Cell 编码长度"""
        return config.NUM_NODES * config.EDGES_PER_NODE * 2


class Individual:
    """
    个体类，表示一个网络架构候选解
    
    包含:
    - normal_cell: Normal Cell 编码
    - reduction_cell: Reduction Cell 编码
    """
    _id_counter = 0
    
    def __init__(self, normal_cell: CellEncoding = None, reduction_cell: CellEncoding = None):
        Individual._id_counter += 1
        self.id = Individual._id_counter
        
        self.normal_cell = normal_cell if normal_cell is not None else CellEncoding()
        self.reduction_cell = reduction_cell if reduction_cell is not None else CellEncoding()
        
        # 评估属性
        self.fitness = None
        self.param_count = None
        self.accuracy = None
    
    def copy(self) -> 'Individual':
        """深拷贝个体"""
        new_ind = Individual(
            normal_cell=self.normal_cell.copy(),
            reduction_cell=self.reduction_cell.copy()
        )
        new_ind.fitness = self.fitness
        new_ind.param_count = self.param_count
        new_ind.accuracy = self.accuracy
        return new_ind
    
    @classmethod
    def update_id_counter(cls, max_id: int):
        """更新 ID 计数器，确保新个体 ID 不与已有个体冲突"""
        if max_id >= cls._id_counter:
            cls._id_counter = max_id + 1
    
    def to_dict(self) -> dict:
        """转换为字典格式 (用于序列化)"""
        return {
            'id': self.id,
            'normal_cell': self.normal_cell.to_list(),
            'reduction_cell': self.reduction_cell.to_list(),
            'fitness': self.fitness,
            'param_count': self.param_count,
            'accuracy': self.accuracy
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Individual':
        """从字典格式创建个体"""
        ind = cls(
            normal_cell=CellEncoding.from_list(data['normal_cell']),
            reduction_cell=CellEncoding.from_list(data['reduction_cell'])
        )
        ind.id = data.get('id', ind.id)
        ind.fitness = data.get('fitness')
        ind.param_count = data.get('param_count')
        ind.accuracy = data.get('accuracy')
        return ind
    
    def validate(self) -> bool:
        """验证个体有效性"""
        return self.normal_cell.validate() and self.reduction_cell.validate()
    
    def __repr__(self):
        return (f"Individual(id={self.id}, fitness={self.fitness}, "
                f"params={self.param_count}, acc={self.accuracy})")


class Encoder:
    """
    编码器工具类
    提供静态方法用于编码操作
    """
    
    @staticmethod
    def print_architecture(individual: Individual):
        """打印架构信息"""
        print(f"\n{'='*60}")
        print(f"Individual {individual.id}")
        print(f"{'='*60}")
        print(f"Fitness: {individual.fitness}")
        print(f"Param Count: {individual.param_count}")
        print(f"Accuracy: {individual.accuracy}")
        print(f"\n--- Normal Cell ---")
        print(individual.normal_cell)
        print(f"\n--- Reduction Cell ---")
        print(individual.reduction_cell)
        print(f"\n{'='*60}\n")
    
    @staticmethod
    def get_genotype(individual: Individual) -> dict:
        """
        获取基因型 (用于与 DARTS 论文对比)
        
        返回格式与 DARTS 原始代码兼容的基因型表示
        """
        def cell_to_genotype(cell: CellEncoding) -> List[Tuple[str, int]]:
            genotype = []
            for node_idx, node_edges in enumerate(cell.edges):
                for edge in node_edges:
                    op_name = config.OPERATIONS[edge.op_id]
                    genotype.append((op_name, edge.source))
            return genotype
        
        return {
            'normal': cell_to_genotype(individual.normal_cell),
            'normal_concat': list(range(2, 2 + config.NUM_NODES)),
            'reduce': cell_to_genotype(individual.reduction_cell),
            'reduce_concat': list(range(2, 2 + config.NUM_NODES))
        }
