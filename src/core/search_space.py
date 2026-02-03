# -*- coding: utf-8 -*-
"""
DARTS 搜索空间模块
定义 DARTS 搜索空间和随机生成逻辑
"""
import random
from typing import List, Optional
from configuration.config import config
from core.encoding import CellEncoding, Individual, Edge, DuplicateChecker
from utils.logger import logger


class DARTSSearchSpace:
    """
    DARTS 搜索空间定义类
    
    搜索空间包含:
    - Cell 拓扑结构 (DAG)
    - 边上的操作类型
    """
    
    def __init__(self):
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        self.operations = config.OPERATIONS
        self.num_operations = len(self.operations)
    
    def get_valid_sources(self, node_idx: int) -> List[int]:
        """
        获取指定节点的有效输入来源
        
        Args:
            node_idx: 中间节点索引 (0 到 NUM_NODES-1)
        
        Returns:
            有效来源列表 [Input_0, Input_1, Node_0, ..., Node_{node_idx-1}]
        """
        # Input_0 = 0, Input_1 = 1, Node_0 = 2, Node_1 = 3, ...
        return list(range(2 + node_idx))
    
    def sample_edge(self, node_idx: int) -> Edge:
        """
        随机采样一条边
        
        Args:
            node_idx: 目标节点索引
        
        Returns:
            随机生成的边
        """
        valid_sources = self.get_valid_sources(node_idx)
        source = random.choice(valid_sources)
        op_id = random.randint(0, self.num_operations - 1)
        return Edge(source=source, op_id=op_id)
    
    def sample_cell(self) -> CellEncoding:
        """
        随机采样一个 Cell 结构
        
        Returns:
            随机生成的 Cell 编码
        """
        edges = []
        for node_idx in range(self.num_nodes):
            node_edges = [self.sample_edge(node_idx) for _ in range(self.edges_per_node)]
            edges.append(node_edges)
        return CellEncoding(edges=edges)
    
    def sample_individual(self) -> Individual:
        """
        采样一个完整个体 (包含 Normal Cell 和 Reduction Cell)
        
        Returns:
            随机生成的个体
        """
        normal_cell = self.sample_cell()
        reduction_cell = self.sample_cell()
        return Individual(normal_cell=normal_cell, reduction_cell=reduction_cell)
    
    def sample_operation(self) -> int:
        """随机采样一个操作 ID"""
        return random.randint(0, self.num_operations - 1)
    
    def sample_source(self, node_idx: int) -> int:
        """随机采样一个有效来源"""
        return random.choice(self.get_valid_sources(node_idx))
    
    def get_operation_name(self, op_id: int) -> str:
        """获取操作名称"""
        if 0 <= op_id < len(self.operations):
            return self.operations[op_id]
        return f"unknown_op_{op_id}"
    
    def get_operation_id(self, op_name: str) -> int:
        """获取操作 ID"""
        try:
            return self.operations.index(op_name)
        except ValueError:
            return -1


class PopulationInitializer:
    """
    种群初始化器
    """
    
    def __init__(self, search_space: DARTSSearchSpace):
        self.search_space = search_space
    
    def create_valid_individual(self, max_attempts: int = 1000) -> Optional[Individual]:
        """
        创建一个有效的个体
        
        DARTS Cell 编码总是有效的，因为采样时已保证约束
        
        Args:
            max_attempts: 最大尝试次数 (保留参数以兼容接口)
        
        Returns:
            有效的个体
        """
        individual = self.search_space.sample_individual()
        
        if individual.validate():
            return individual
        
        # 理论上不应该到达这里
        logger.warning("Generated individual failed validation, retrying...")
        for _ in range(max_attempts):
            individual = self.search_space.sample_individual()
            if individual.validate():
                return individual
        
        logger.error(f"Failed to create valid individual after {max_attempts} attempts")
        return None
    
    def create_unique_individual(self, existing_population: List[Individual], 
                                max_attempts: int = None) -> Optional[Individual]:
        """
        创建一个不与现有种群重复的个体
        
        Args:
            existing_population: 现有种群列表
            max_attempts: 最大尝试次数，默认使用配置值
        
        Returns:
            不重复的有效个体，如果超过最大尝试次数则返回重复的个体
        """
        
        if max_attempts is None:
            max_attempts = config.MAX_INIT_DUPLICATE_ATTEMPTS
        
        # 快速判重：预先计算编码集合
        encoding_set = DuplicateChecker.get_encoding_set(existing_population)
        
        for attempt in range(max_attempts):
            individual = self.create_valid_individual()
            
            if individual is None:
                continue
            
            # 检查是否重复
            if not DuplicateChecker.is_duplicate_fast(individual, encoding_set):
                if attempt > 0:
                    logger.info(f"Created unique individual after {attempt + 1} attempts")
                return individual
        
        # 超过最大尝试次数，返回最后一个个体（可能重复）
        logger.warning(
            f"Failed to create unique individual after {max_attempts} attempts; "
            "returning the last valid individual (may be duplicate)"
        )
        # 确保返回一个有效个体
        return self.create_valid_individual()


# Global instances
search_space = DARTSSearchSpace()
population_initializer = PopulationInitializer(search_space)
