# -*- coding: utf-8 -*-
"""
DARTS Cell 变异算子模块
实现修改边操作和修改节点输入来源的变异操作
"""
import random
import copy
from typing import List, Tuple
from configuration.config import config
from core.encoding import CellEncoding, Individual, Edge
from core.search_space import search_space
from utils.logger import logger


class MutationOperator:
    """
    DARTS Cell 变异算子
    
    实现两种变异操作:
    1. 修改边操作 (mutate_edge_operation)
    2. 修改节点输入来源 (mutate_edge_source)
    """
    
    def __init__(self):
        self.prob_mutate_normal = config.PROB_MUTATE_NORMAL
        self.prob_mutate_reduction = config.PROB_MUTATE_REDUCTION
        self.prob_mutate_operation = config.PROB_MUTATE_OPERATION
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        self.num_operations = len(config.OPERATIONS)
    
    def mutate_edge_operation(self, cell: CellEncoding) -> CellEncoding:
        """
        变异算子1: 修改一条边代表的操作
        
        随机选择一条边，将其操作替换为另一个操作
        
        Args:
            cell: Cell 编码
        
        Returns:
            变异后的 Cell 编码
        """
        # 随机选择一个节点
        node_idx = random.randint(0, self.num_nodes - 1)
        # 随机选择该节点的一条边
        edge_idx = random.randint(0, self.edges_per_node - 1)
        
        # 获取当前边
        edge = cell.get_edge(node_idx, edge_idx)
        
        # 选择一个不同的操作
        new_op_id = edge.op_id
        attempts = 0
        while new_op_id == edge.op_id and attempts < 10:
            new_op_id = random.randint(0, self.num_operations - 1)
            attempts += 1
        
        # 更新边
        new_edge = Edge(source=edge.source, op_id=new_op_id)
        cell.set_edge(node_idx, edge_idx, new_edge)
        
        return cell
    
    def mutate_edge_source(self, cell: CellEncoding, node_idx: int = None) -> CellEncoding:
        """
        变异算子2: 修改一个节点的一个输入的来源
        
        随机选择一条边，将其输入来源替换为另一个有效来源
        
        Args:
            cell: Cell 编码
            node_idx: 可选，指定要修改的节点索引。如果为 None，则随机选择
        
        Returns:
            变异后的 Cell 编码
        """
        # 选择节点
        if node_idx is None:
            node_idx = random.randint(0, self.num_nodes - 1)
        
        # 随机选择该节点的一条边
        edge_idx = random.randint(0, self.edges_per_node - 1)
        
        # 获取当前边
        edge = cell.get_edge(node_idx, edge_idx)
        
        # 获取有效来源列表
        valid_sources = cell.get_valid_sources(node_idx)
        
        if len(valid_sources) <= 1:
            # 只有一个有效来源，无法变异
            return cell
        
        # 选择一个不同的来源
        new_source = edge.source
        attempts = 0
        while new_source == edge.source and attempts < 10:
            new_source = random.choice(valid_sources)
            attempts += 1
        
        # 更新边
        new_edge = Edge(source=new_source, op_id=edge.op_id)
        cell.set_edge(node_idx, edge_idx, new_edge)
        
        return cell
    
    def mutate_cell(self, cell: CellEncoding) -> CellEncoding:
        """
        对单个 Cell 进行变异
        
        以相等概率选择变异类型:
        - 修改边操作
        - 修改节点输入来源
        
        Args:
            cell: Cell 编码
        
        Returns:
            变异后的 Cell 编码
        """
        # 深拷贝避免修改原始数据
        cell = cell.copy()
        
        if random.random() < self.prob_mutate_operation:
            cell = self.mutate_edge_operation(cell)
        else:
            cell = self.mutate_edge_source(cell)
        
        return cell
    
    def mutate(self, individual: Individual) -> Individual:
        """
        对个体进行变异
        
        变异策略:
        1. 以概率 PROB_MUTATE_NORMAL 变异 Normal Cell
        2. 以概率 PROB_MUTATE_REDUCTION 变异 Reduction Cell
        
        Args:
            individual: 原始个体
        
        Returns:
            变异后的新个体
        """
        # 深拷贝
        new_individual = individual.copy()
        mutation_applied = False
        
        # 变异 Normal Cell
        if random.random() < self.prob_mutate_normal:
            new_individual.normal_cell = self.mutate_cell(new_individual.normal_cell)
            mutation_applied = True
        
        # 变异 Reduction Cell
        if random.random() < self.prob_mutate_reduction:
            new_individual.reduction_cell = self.mutate_cell(new_individual.reduction_cell)
            mutation_applied = True
        
        # 如果没有应用任何变异，强制变异其中一个 Cell
        if not mutation_applied:
            if random.random() < 0.5:
                new_individual.normal_cell = self.mutate_cell(new_individual.normal_cell)
            else:
                new_individual.reduction_cell = self.mutate_cell(new_individual.reduction_cell)
        
        # 重置适应度
        new_individual.fitness = None
        new_individual.param_count = None
        new_individual.accuracy = None
        
        logger.log_mutation("mutate", individual.id, new_individual.id)
        
        return new_individual


class SelectionOperator:
    """
    选择算子 (锦标赛选择)
    """
    
    def tournament_selection(self, population: List[Individual], 
                             tournament_size: int = None, 
                             num_winners: int = None) -> List[Individual]:
        """
        锦标赛选择
        
        Args:
            population: 种群列表
            tournament_size: 锦标赛大小
            num_winners: 胜者数量
        
        Returns:
            胜者列表
        """
        if tournament_size is None:
            tournament_size = config.TOURNAMENT_SIZE
        if num_winners is None:
            num_winners = config.TOURNAMENT_WINNERS
        
        tournament_size = min(tournament_size, len(population))
        
        # 随机选择参与者
        competitors = random.sample(list(population), tournament_size)
        
        # 按 fitness 排序 (fitness 越小越好)
        sorted_competitors = sorted(
            competitors, 
            key=lambda x: x.fitness if x.fitness is not None else float('inf'),
            reverse=False
        )
        
        return sorted_competitors[:num_winners]


class CrossoverOperator:
    """
    交叉算子
    
    Cell 级别交叉: 交换 Normal Cell 或 Reduction Cell
    """
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Cell 级别交叉
        
        策略:
        - 以 50% 概率从 parent1 或 parent2 选择 Normal Cell
        - 以 50% 概率从 parent1 或 parent2 选择 Reduction Cell
        
        Args:
            parent1: 父代1
            parent2: 父代2
        
        Returns:
            子代个体
        """
        child = Individual()
        
        # 选择 Normal Cell
        if random.random() < 0.5:
            child.normal_cell = parent1.normal_cell.copy()
        else:
            child.normal_cell = parent2.normal_cell.copy()
        
        # 选择 Reduction Cell
        if random.random() < 0.5:
            child.reduction_cell = parent1.reduction_cell.copy()
        else:
            child.reduction_cell = parent2.reduction_cell.copy()
        
        return child
    
    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        均匀交叉 (生成两个子代)
        
        Args:
            parent1: 父代1
            parent2: 父代2
        
        Returns:
            两个子代个体
        """
        child1 = Individual()
        child2 = Individual()
        
        # Normal Cell 交叉
        if random.random() < 0.5:
            child1.normal_cell = parent1.normal_cell.copy()
            child2.normal_cell = parent2.normal_cell.copy()
        else:
            child1.normal_cell = parent2.normal_cell.copy()
            child2.normal_cell = parent1.normal_cell.copy()
        
        # Reduction Cell 交叉
        if random.random() < 0.5:
            child1.reduction_cell = parent1.reduction_cell.copy()
            child2.reduction_cell = parent2.reduction_cell.copy()
        else:
            child1.reduction_cell = parent2.reduction_cell.copy()
            child2.reduction_cell = parent1.reduction_cell.copy()
        
        return child1, child2
    
    def edge_level_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        边级别交叉 (更细粒度的交叉)
        
        对于每条边，以 50% 概率从 parent1 或 parent2 选择
        
        Args:
            parent1: 父代1
            parent2: 父代2
        
        Returns:
            子代个体
        """
        child = Individual()
        
        # Normal Cell 边级别交叉
        child.normal_cell = self._crossover_cell_edges(
            parent1.normal_cell, parent2.normal_cell
        )
        
        # Reduction Cell 边级别交叉
        child.reduction_cell = self._crossover_cell_edges(
            parent1.reduction_cell, parent2.reduction_cell
        )
        
        return child
    
    def _crossover_cell_edges(self, cell1: CellEncoding, cell2: CellEncoding) -> CellEncoding:
        """
        对两个 Cell 进行边级别交叉
        """
        num_nodes = config.NUM_NODES
        edges_per_node = config.EDGES_PER_NODE
        
        new_edges = []
        for node_idx in range(num_nodes):
            node_edges = []
            for edge_idx in range(edges_per_node):
                if random.random() < 0.5:
                    edge = cell1.get_edge(node_idx, edge_idx).copy()
                else:
                    edge = cell2.get_edge(node_idx, edge_idx).copy()
                node_edges.append(edge)
            new_edges.append(node_edges)
        
        new_cell = CellEncoding.__new__(CellEncoding)
        new_cell.num_nodes = num_nodes
        new_cell.edges_per_node = edges_per_node
        new_cell.edges = new_edges
        
        return new_cell


# 全局实例
mutation_operator = MutationOperator()
selection_operator = SelectionOperator()
crossover_operator = CrossoverOperator()
