import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.config import config

def calculate_search_space():
    # 基础参数选项数量 (不包含受约束的参数: out_channels, skip_type)
    # groups(3) * pool_type(2) * pool_stride(2) * senet(2) * activation(1) * dropout(2) * kernel(2) * expansion(2)
    # = 192
    base_complexity = len(config.GROUP_OPTIONS) * \
                      len(config.POOL_TYPE_OPTIONS) * \
                      len(config.POOL_STRIDE_OPTIONS) * \
                      len(config.SENET_OPTIONS) * \
                      len(config.ACTIVATION_OPTIONS) * \
                      len(config.DROPOUT_OPTIONS) * \
                      len(config.KERNEL_SIZE_OPTIONS) * \
                      len(config.EXPANSION_OPTIONS)

    total_space = 0
    
    # 遍历所有可能的 Unit 数量 (3 到 5)
    for unit_num in range(config.MIN_UNIT_NUM, config.MAX_UNIT_NUM + 1):
        
        unit_complexities = []
        
        for u_idx in range(unit_num):
            # 当前 Unit 的总复杂度 (即该 Unit 所有可能的内部 Block 结构的组合总数)
            current_unit_complexity = 0
            
            # 遍历当前 Unit 可能的 Block 数量 (2 到 4)
            for block_num in range(config.MIN_BLOCK_NUM, config.MAX_BLOCK_NUM + 1):
                # 对于特定的 block_num，计算该 Unit 的具体复杂度
                # 是各个 Block 复杂度的乘积
                config_complexity = 1
                
                for b_idx in range(block_num):
                    # === 计算单个 Block 的可能性 ===
                    
                    # 1. 基础复杂度
                    block_opts = base_complexity
                    
                    # 2. Channel 选项 (受 u_idx 约束)
                    # Constraint: 64 only in first unit (u_idx == 0)
                    # Constraint: 1024 only in last unit (u_idx == unit_num - 1)
                    valid_channels = 0
                    for ch in config.CHANNEL_OPTIONS:
                        if u_idx > 0 and ch == 64:
                            continue
                        if u_idx < unit_num - 1 and ch == 1024:
                            continue
                        valid_channels += 1
                    block_opts *= valid_channels
                    
                    # 3. Skip Type 选项 (受 b_idx 约束)
                    # Constraint: Concat (1) only allowed at last block of unit
                    valid_skips = 0
                    is_last_block = (b_idx == block_num - 1)
                    for skip in config.SKIP_TYPE_OPTIONS:
                        if skip == config.SKIP_TYPE_CONCAT and not is_last_block:
                            continue
                        valid_skips += 1
                    block_opts *= valid_skips
                    
                    config_complexity *= block_opts
                
                current_unit_complexity += config_complexity
            
            unit_complexities.append(current_unit_complexity)
        
        # 计算当前 Unit 数量下的总架构数
        structure_complexity = 1
        for uc in unit_complexities:
            structure_complexity *= uc
            
        total_space += structure_complexity

    return total_space

if __name__ == "__main__":
    try:
        size = calculate_search_space()
        print(f"Approximate Search Space Size: {size:.4e}")
        # Estimating Feature Size Constraint Reduction
        # Roughly, valid strides constitute ~2% of the space for deep networks, 
        # but let's just report the structural space size first.
    except Exception as e:
        print(f"Error: {e}")
