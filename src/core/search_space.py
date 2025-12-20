# -*- coding: utf-8 -*-
"""
搜索空间模块
定义搜索空间和随机生成逻辑
"""
import random
from typing import List, Optional
from configuration.config import config
from core.encoding import Encoder, Individual, BlockParams
from utils.logger import logger

class SearchSpace:
    """
    搜索空间定义类
    """
    def __init__(self):
        self.min_unit_num = config.MIN_UNIT_NUM
        self.max_unit_num = config.MAX_UNIT_NUM
        self.min_block_num = config.MIN_BLOCK_NUM
        self.max_block_num = config.MAX_BLOCK_NUM
        self.channel_options = config.CHANNEL_OPTIONS
        self.group_options = config.GROUP_OPTIONS
        self.pool_type_options = config.POOL_TYPE_OPTIONS
        self.pool_stride_options = config.POOL_STRIDE_OPTIONS
        self.senet_options = config.SENET_OPTIONS
    
    def sample_unit_num(self) -> int:
        return random.randint(self.min_unit_num, self.max_unit_num)
    
    def sample_block_num(self) -> int:
        return random.randint(self.min_block_num, self.max_block_num)
    
    def sample_channel(self) -> int:
        return random.choice(self.channel_options)
    
    def sample_groups(self, max_groups: int = None) -> int:
        if max_groups is None:
            return random.choice(self.group_options)
        valid_groups = [g for g in self.group_options if g <= max_groups]
        return random.choice(valid_groups) if valid_groups else 1
    
    def sample_pool_type(self) -> int:
        return random.choice(self.pool_type_options)
    
    def sample_pool_stride(self) -> int:
        return random.choice(self.pool_stride_options)
    
    def sample_senet(self) -> int:
        return random.choice(self.senet_options)
    
    def sample_block_params(self) -> BlockParams:
        out_channels = self.sample_channel()
        groups = self.sample_groups(out_channels)
        pool_type = self.sample_pool_type()
        pool_stride = self.sample_pool_stride()
        has_senet = self.sample_senet()
        return BlockParams(out_channels, groups, pool_type, pool_stride, has_senet)

class PopulationInitializer:
    """
    种群初始化器
    """
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
    
    def create_valid_individual(self) -> Optional[Individual]:
        while(True):
            encoding = self._create_constrained_encoding()
            if Encoder.validate_encoding(encoding):
                return Individual(encoding)

            
    
    def _create_constrained_encoding(self) -> List[int]:
        max_downsampling = Encoder.get_max_downsampling()
        unit_num = self.search_space.sample_unit_num()
        encoding = [unit_num]
        block_nums = []
        
        for _ in range(unit_num):
            block_num = self.search_space.sample_block_num()
            block_nums.append(block_num)
            encoding.append(block_num)
        
        downsampling_count = 0
        
        for block_num in block_nums:
            for _ in range(block_num):
                block_params = self.search_space.sample_block_params()
                
                if downsampling_count >= max_downsampling and block_params.pool_stride == 2:
                    block_params.pool_stride = 1
                elif block_params.pool_stride == 2:
                    downsampling_count += 1
                
                encoding.extend(block_params.to_list())
        
        return encoding

# Global instances
search_space = SearchSpace()
population_initializer = PopulationInitializer(search_space)
