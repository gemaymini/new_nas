# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 编码模块
实现变长编码策略
"""
import random
import copy
from typing import List, Tuple, Optional
from configuration.config import config

class BlockParams:
    """
    Block参数封装类
    """
    def __init__(self, out_channels: int, groups: int, pool_type: int, 
                 pool_stride: int, has_senet: int):
        self.out_channels = out_channels
        self.groups = groups
        self.pool_type = pool_type
        self.pool_stride = pool_stride
        self.has_senet = has_senet
    
    def to_list(self) -> List[int]:
        return [self.out_channels, self.groups, self.pool_type, 
                self.pool_stride, self.has_senet]
    
    @classmethod
    def from_list(cls, params: List[int]) -> 'BlockParams':
        return cls(params[0], params[1], params[2], params[3], params[4])
    
    def __repr__(self):
        return (f"BlockParams(out_ch={self.out_channels}, groups={self.groups}, "
                f"pool_type={self.pool_type}, pool_stride={self.pool_stride}, "
                f"has_senet={self.has_senet})")

class Individual:
    """
    个体类，表示一个网络架构候选解
    """
    def __init__(self, encoding: Optional[List[int]] = None):
        self.id = None       
        self.encoding = encoding if encoding is not None else []
        self.quick_score=0
        # 评估属性
        self.fitness = None
        
    def copy(self) -> 'Individual':
        new_ind = Individual(copy.deepcopy(self.encoding))
        new_ind.fitness = self.fitness
        return new_ind
    
    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness}, encoding_len={len(self.encoding)})"

class Encoder:
    """
    编码器类
    """
    @staticmethod
    def random_block_params() -> BlockParams:
        out_channels = random.choice(config.CHANNEL_OPTIONS)
        groups = random.choice(config.GROUP_OPTIONS)
        while groups > out_channels:
            groups = random.choice(config.GROUP_OPTIONS)
        pool_type = random.choice(config.POOL_TYPE_OPTIONS)
        pool_stride = random.choice(config.POOL_STRIDE_OPTIONS)
        has_senet = random.choice(config.SENET_OPTIONS)
        
        return BlockParams(out_channels, groups, pool_type, pool_stride, has_senet)
    
    @staticmethod
    def create_random_encoding() -> List[int]:
        unit_num = random.randint(config.MIN_UNIT_NUM, config.MAX_UNIT_NUM)
        encoding = [unit_num]
        
        block_nums = []
        for _ in range(unit_num):
            block_num = random.randint(config.MIN_BLOCK_NUM, config.MAX_BLOCK_NUM)
            block_nums.append(block_num)
            encoding.append(block_num)
        
        for block_num in block_nums:
            for _ in range(block_num):
                block_params = Encoder.random_block_params()
                encoding.extend(block_params.to_list())
        
        return encoding
    
    @staticmethod
    def decode(encoding: List[int]) -> Tuple[int, List[int], List[List[BlockParams]]]:
        if not encoding:
            raise ValueError("Encoding is empty")
        
        unit_num = encoding[0]
        block_nums = encoding[1:1+unit_num]
        
        block_params_list = []
        idx = 1 + unit_num
        
        for block_num in block_nums:
            unit_blocks = []
            for _ in range(block_num):
                params = encoding[idx:idx+5]
                if len(params) < 5:
                    raise ValueError(f"Incomplete block params at index {idx}")
                block_params = BlockParams.from_list(params)
                unit_blocks.append(block_params)
                idx += 5
            block_params_list.append(unit_blocks)
        
        return unit_num, block_nums, block_params_list
    
    @staticmethod
    def encode(unit_num: int, block_nums: List[int], 
               block_params_list: List[List[BlockParams]]) -> List[int]:
        encoding = [unit_num]
        encoding.extend(block_nums)
        
        for unit_blocks in block_params_list:
            for block_params in unit_blocks:
                encoding.extend(block_params.to_list())
        
        return encoding
    
    @staticmethod
    def validate_encoding(encoding: List[int]) -> bool:
        try:
            unit_num, block_nums, block_params_list = Encoder.decode(encoding)
            
            if not (config.MIN_UNIT_NUM <= unit_num <= config.MAX_UNIT_NUM):
                return False
            
            for block_num in block_nums:
                if not (config.MIN_BLOCK_NUM <= block_num <= config.MAX_BLOCK_NUM):
                    return False
            
            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    if bp.out_channels not in config.CHANNEL_OPTIONS: return False
                    if bp.groups not in config.GROUP_OPTIONS: return False
                    if bp.groups > bp.out_channels: return False
                    if bp.pool_type not in config.POOL_TYPE_OPTIONS: return False
                    if bp.pool_stride not in config.POOL_STRIDE_OPTIONS: return False
                    if bp.has_senet not in config.SENET_OPTIONS: return False
            
            if not Encoder.validate_feature_size(encoding):
                return False
            
            return True
        except Exception:
            return False

    @staticmethod
    def validate_feature_size(encoding: List[int], input_size: int = None) -> bool:
        if input_size is None:
            input_size = config.INPUT_IMAGE_SIZE
        
        try:
            _, _, block_params_list = Encoder.decode(encoding)
            current_size = input_size
            
            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    if bp.pool_stride == 2:
                        current_size = (current_size + 1) // 2
                    if current_size < config.MIN_FEATURE_SIZE:
                        return False
            return current_size >= config.MIN_FEATURE_SIZE
        except Exception:
            return False
            
    @staticmethod
    def get_max_downsampling(input_size: int = None) -> int:
        if input_size is None:
            input_size = config.INPUT_IMAGE_SIZE
        import math
        return int(math.log2(input_size / config.MIN_FEATURE_SIZE))

    @staticmethod
    def print_architecture(encoding: List[int]):
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)
        print(f"\n{'='*60}")
        print(f"Network Architecture")
        print(f"{'='*60}")
        print(f"Number of Units: {unit_num}")
        print(f"Blocks per Unit: {block_nums}")
        print(f"{'-'*60}")
        for i, unit_blocks in enumerate(block_params_list):
            print(f"\nUnit {i+1} ({len(unit_blocks)} blocks):")
            for j, bp in enumerate(unit_blocks):
                pool_type_str = "MaxPool" if bp.pool_type == 0 else "AvgPool"
                senet_str = "Yes" if bp.has_senet == 1 else "No"
                print(f"  Block {j+1}: out_ch={bp.out_channels}, groups={bp.groups}, "
                      f"pool={pool_type_str}, stride={bp.pool_stride}, SENet={senet_str}")
        print(f"\n{'='*60}\n")
