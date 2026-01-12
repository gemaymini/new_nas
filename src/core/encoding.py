# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 编码模块
实现变长编码策略
"""
import random
import copy
from typing import List, Tuple, Optional
from configuration.config import config

# Block参数数量常量（用于编解码）
BLOCK_PARAM_COUNT = 9

class BlockParams:
    """
    Block参数封装类
    """
    def __init__(self, out_channels: int, groups: int, pool_type: int, 
                 pool_stride: int, has_senet: int, activation_type: int = 0,
                 dropout_rate: float = 0.0, skip_type: int = 0, kernel_size: int = 3):
        self.out_channels = out_channels
        self.groups = groups
        self.pool_type = pool_type
        self.pool_stride = pool_stride
        self.has_senet = has_senet
        self.activation_type = activation_type  # 0=ReLU, 1=SiLU, 2=GELU
        self.dropout_rate = dropout_rate        # Dropout率
        self.skip_type = skip_type              # 0=add, 1=concat, 2=none
        self.kernel_size = kernel_size          # 卷积核大小: 3, 5, 7
    
    def to_list(self) -> List:
        dropout_encoded = int(self.dropout_rate * 100)
        return [self.out_channels, self.groups, self.pool_type, 
                self.pool_stride, self.has_senet, self.activation_type,
                dropout_encoded, self.skip_type, self.kernel_size]
    
    @classmethod
    def from_list(cls, params: List) -> 'BlockParams':
        dropout_rate = params[6] / 100.0  # 解码dropout率
        return cls(params[0], params[1], params[2], params[3], params[4],
                   params[5], dropout_rate, params[7], params[8])
    
    def __repr__(self):
        activation_names = {0: 'ReLU', 1: 'SiLU', 2: 'GELU'}
        skip_names = {0: 'add', 1: 'concat', 2: 'none'}
        return (f"BlockParams(out_ch={self.out_channels}, groups={self.groups}, "
                f"pool_type={self.pool_type}, pool_stride={self.pool_stride}, "
                f"has_senet={self.has_senet}, activation={activation_names.get(self.activation_type, 'ReLU')}, "
                f"dropout={self.dropout_rate}, skip={skip_names.get(self.skip_type, 'add')}, "
                f"kernel_size={self.kernel_size})")

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
        # 记录导致本个体的交叉/变异历程
        self.op_history = []
        
    def copy(self) -> 'Individual':
        new_ind = Individual(copy.deepcopy(self.encoding))
        new_ind.fitness = self.fitness
        new_ind.quick_score = self.quick_score
        new_ind.op_history = copy.deepcopy(self.op_history)
        return new_ind
    
    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness}, encoding_len={len(self.encoding)})"

class Encoder:
    """
    编码器类
    """
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
                params = encoding[idx:idx+BLOCK_PARAM_COUNT]
                if len(params) < BLOCK_PARAM_COUNT:
                    raise ValueError(f"Incomplete block params at index {idx}")
                block_params = BlockParams.from_list(params)
                unit_blocks.append(block_params)
                idx += BLOCK_PARAM_COUNT
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
                print("Unit数量不合法")
                return False
            
            for block_num in block_nums:
                if not (config.MIN_BLOCK_NUM <= block_num <= config.MAX_BLOCK_NUM):
                    print("Block数量不合法")
                    return False
            
            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    if bp.out_channels not in config.CHANNEL_OPTIONS: return False
                    if bp.groups not in config.GROUP_OPTIONS: return False
                    if bp.pool_type not in config.POOL_TYPE_OPTIONS: return False
                    if bp.pool_stride not in config.POOL_STRIDE_OPTIONS: return False
                    if bp.has_senet not in config.SENET_OPTIONS: return False
                    if bp.activation_type not in config.ACTIVATION_OPTIONS: return False
                    if bp.dropout_rate not in config.DROPOUT_OPTIONS: return False
                    if bp.skip_type not in config.SKIP_TYPE_OPTIONS: return False
                    if bp.kernel_size not in config.KERNEL_SIZE_OPTIONS: return False
            
            if not Encoder.validate_feature_size(encoding):
                print("特征图尺寸过小")
                return False
            
            # 验证通道数不会爆炸（特别是 concat 模式）
            if not Encoder.validate_channel_count(encoding):
                print("通道数爆炸")
                return False
            
            return True
        except Exception:
            return False

    @staticmethod
    def validate_channel_count(encoding: List[int], init_channels: int = None) -> bool:
        """验证网络通道数不会超过最大限制"""
        if init_channels is None:
            init_channels = config.INIT_CONV_OUT_CHANNELS
        
        try:
            _, _, block_params_list = Encoder.decode(encoding)
            current_channels = init_channels
            
            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    out_channels = bp.out_channels * config.EXPANSION
                    if bp.skip_type == 1:  # concat
                        final_channels = out_channels + current_channels
                    else:
                        final_channels = out_channels
                    
                    if final_channels > config.MAX_CHANNELS:
                        return False
                    current_channels = final_channels
            
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
        
        # 尝试计算参数量
        param_count_str = "N/A"
        try:
            # 延迟导入以避免循环依赖
            from models.network import NetworkBuilder
            param_count = NetworkBuilder.calculate_param_count(encoding)
            param_count_str = f"{param_count:,}"
        except Exception as e:
            param_count_str = f"Error calculating params: {e}"

        print(f"\n{'='*60}")
        print(f"Network Architecture")
        print(f"{'='*60}")
        print(f"Number of Units: {unit_num}")
        print(f"Blocks per Unit: {block_nums}")
        print(f"Total Parameters: {param_count_str}")
        print(f"{'-'*60}")
        for i, unit_blocks in enumerate(block_params_list):
            print(f"\nUnit {i+1} ({len(unit_blocks)} blocks):")
            for j, bp in enumerate(unit_blocks):
                pool_type_str = "MaxPool" if bp.pool_type == 0 else "AvgPool"
                senet_str = "Yes" if bp.has_senet == 1 else "No"
                activation_names = {0: 'ReLU', 1: 'SiLU', 2: 'GELU'}
                skip_names = {0: 'add', 1: 'concat', 2: 'none'}
                activation_str = activation_names.get(bp.activation_type, 'ReLU')
                skip_str = skip_names.get(bp.skip_type, 'add')
                print(f"  Block {j+1}: out_ch={bp.out_channels}, groups={bp.groups}, "
                      f"pool={pool_type_str}, stride={bp.pool_stride}, SENet={senet_str}, "
                      f"act={activation_str}, dropout={bp.dropout_rate}, skip={skip_str}, "
                      f"kernel={bp.kernel_size}")
        print(f"\n{'='*60}\n")
