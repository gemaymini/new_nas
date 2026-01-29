# -*- coding: utf-8 -*-
"""
DARTS-based 神经网络架构搜索算法 - 配置文件
包含所有超参数配置
"""
import random


class Config:
    """
    配置类，包含所有超参数
    """
    
    # ==================== DARTS Cell 结构参数 ====================
    NUM_NODES = 4                    # Cell 中的中间节点数
    EDGES_PER_NODE = 2               # 每个节点的输入边数 (论文默认为 2)
    
    # DARTS 操作集合 (8 种标准操作)
    OPERATIONS = [
        'zero',          # 0: 无连接
        'skip_connect',  # 1: 恒等映射
        'sep_conv_3x3',  # 2: 3×3 深度可分离卷积
        'sep_conv_5x5',  # 3: 5×5 深度可分离卷积
        'dil_conv_3x3',  # 4: 3×3 空洞卷积 (dilation=2)
        'dil_conv_5x5',  # 5: 5×5 空洞卷积 (dilation=2)
        'max_pool_3x3',  # 6: 3×3 最大池化
        'avg_pool_3x3',  # 7: 3×3 平均池化
    ]
    
    # ==================== 网络堆叠参数 ====================
    INIT_CHANNELS = 36               # 初始通道数 (论文默认 36)
    CELLS_PER_STAGE = 6              # 每个 Stage 的 Normal Cell 数量
    NUM_STAGES = 3                   # Stage 数量 (3 个 stage = 2 个 reduction)
    
    # ==================== 进化算法参数 ====================
    POPULATION_SIZE = 200            # 种群大小 (Aging Evolution Queue Size)
    MAX_GEN = 2000                   # 最大进化代数
    TOURNAMENT_SIZE = 5              # 锦标赛选择的样本大小
    TOURNAMENT_WINNERS = 2           # 锦标赛选择的胜者数量
    
    # ==================== 筛选与训练流程参数 ====================
    HISTORY_TOP_N1 = 10              # 第一轮筛选：基于NTK选择Top N1
    SHORT_TRAIN_EPOCHS = 30          # 第一轮筛选：短期训练轮数
    
    HISTORY_TOP_N2 = 5               # 第二轮筛选：基于验证集Acc选择Top N2
    FULL_TRAIN_EPOCHS = 600          # 最终训练：完整训练轮数
    
    # ==================== 交叉/变异参数 ====================
    PROB_CROSSOVER = 0.5             # 交叉概率
    PROB_MUTATION = 0.5              # 变异概率
    
    # DARTS Cell 变异参数
    PROB_MUTATE_NORMAL = 0.5         # 变异 Normal Cell 的概率
    PROB_MUTATE_REDUCTION = 0.5      # 变异 Reduction Cell 的概率
    PROB_MUTATE_OPERATION = 0.5      # 边操作变异 vs 边来源变异
    
    # ==================== NTK评估配置 ====================
    NTK_BATCH_SIZE = 32              
    NTK_INPUT_SIZE = (3, 32, 32)    
    NTK_NUM_CLASSES = 10            
    NTK_PARAM_THRESHOLD = 10000000   # 参数量阈值
    NTK_CELLS_PER_STAGE = 3          # NTK评估时使用的每个Stage的Cell数 (减少深度以避免随机权重的梯度消失)
    
    # ==================== 训练参数 ====================
    DEVICE = 'cuda'                 
    BATCH_SIZE = 256
    LEARNING_RATE = 0.025             
    MOMENTUM = 0.9                  
    WEIGHT_DECAY = 3e-4
    
    # ==================== 早停参数 ====================
    EARLY_STOPPING_PATIENCE = 75     # 早停耐心值：连续多少个 epoch 没有改进就停止
    EARLY_STOPPING_MIN_DELTA = 0.01   # 最小改进阈值：小于此值的改进不算有效改进
    
    # ==================== 最终评估参数 ====================
    FINAL_DATASET = 'cifar10'       
    
    # ==================== 日志参数 ====================
    LOG_DIR = './logs'              
    LOG_LEVEL = 'INFO'              
    SAVE_CHECKPOINT = True          
    CHECKPOINT_DIR = './checkpoints'  
    

    # ==================== 调试参数 ====================
    SAVE_FAILED_INDIVIDUALS = True  
    FAILED_INDIVIDUALS_DIR = './failed_individuals'  
    
    # ==================== 架构约束参数 ====================
    MIN_FEATURE_SIZE = 1            
    INPUT_IMAGE_SIZE = 32
    
    # ==================== 其他参数 ====================
    RANDOM_SEED = random.randint(0, 2**32-1)
    NUM_WORKERS = 8                 

    def get_search_space_summary(self) -> str:
        """获取搜索空间的摘要字符串"""
        return (
            f"DARTS Search Space Summary:\n"
            f"  - Nodes per Cell: {self.NUM_NODES}\n"
            f"  - Edges per Node: {self.EDGES_PER_NODE}\n"
            f"  - Operations: {len(self.OPERATIONS)}\n"
            f"  - Initial Channels: {self.INIT_CHANNELS}\n"
            f"  - Cells per Stage: {self.CELLS_PER_STAGE}\n"
            f"  - Stages: {self.NUM_STAGES}\n"
            f"  - Total Cells: {self.NUM_STAGES * self.CELLS_PER_STAGE + (self.NUM_STAGES - 1)}"
        )


# 全局配置实例
config = Config()
