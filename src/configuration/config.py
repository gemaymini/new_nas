# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 配置文件
包含所有超参数配置
"""
import os
import random

class Config:
    """
    配置类，包含所有超参数
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    
    # ==================== 进化算法参数 ====================
    POPULATION_SIZE =100         # 种群大小 (Aging Evolution Queue Size)
    MAX_GEN = 3000               # 最大进化代数 (Total number of individuals to evaluate in search)
    TOURNAMENT_SIZE = 5            # 锦标赛选择的样本大小 (Sample Size)
    TOURNAMENT_WINNERS = 2          # 锦标赛选择的胜者数量 (Parent Size)
    
    # ==================== 筛选与训练流程参数 ====================
    HISTORY_TOP_N1 = 5             # 第一轮筛选：基于NTK选择Top N1
    SHORT_TRAIN_EPOCHS = 50         # 第一轮筛选：短期训练轮数
    
    HISTORY_TOP_N2 =  1             # 第二轮筛选：基于验证集Acc选择Top N2
    FULL_TRAIN_EPOCHS = 500         # 最终训练：完整训练轮数
    
    # ==================== 交叉算子参数 ====================
    PROB_CROSSOVER = 0.5            # 交叉概率 (0.8)
    PROB_MUTATION = 0.5            # 变异概率 (0.1)
    

    # ==================== 搜索空间参数 ====================
    MIN_UNIT_NUM = 3                # 最小unit数量
    MAX_UNIT_NUM = 6               # 最大unit数量
    
    MIN_BLOCK_NUM = 2               # 每个unit最小block数量
    MAX_BLOCK_NUM = 6               # 每个unit最大block数量
    
    CHANNEL_OPTIONS = [16,32, 64, 128, 256] 
    GROUP_OPTIONS = [1, 2, 4, 8, 16]
    POOL_TYPE_OPTIONS = [0, 1]
    POOL_STRIDE_OPTIONS = [1, 2]
    SENET_OPTIONS = [0, 1]
    
    # 新增搜索空间参数
    # 激活函数类型: 0=ReLU, 1=SiLU, 2=GELU
    ACTIVATION_OPTIONS = [0, 1, 2]
    # Dropout率选项
    DROPOUT_OPTIONS = [0.0, 0.1, 0.2, 0.3]
    # 跳跃连接类型: 0=add, 1=concat, 2=none
    SKIP_TYPE_OPTIONS = [0, 1, 2]
    # 卷积核大小
    KERNEL_SIZE_OPTIONS = [3, 5, 7]
    
    # Block扩展参数 (输出通道 = 中间通道 × EXPANSION)
    # EXPANSION=1 时与原模型一致，EXPANSION=2 时类似ResNeXt
    EXPANSION = 2
    
    # 初始卷积层参数
    INIT_CONV_OUT_CHANNELS = 64   
    INIT_CONV_KERNEL_SIZE = 3       
    INIT_CONV_STRIDE = 1            
    INIT_CONV_PADDING = 1           
    
    # ==================== 变异概率参数 ====================
    PROB_SWAP_BLOCKS = 0.8          
    PROB_SWAP_UNITS = 0.8          
    PROB_ADD_UNIT = 0.4             
    PROB_ADD_BLOCK = 0.6           
    PROB_DELETE_UNIT = 0.4          
    PROB_DELETE_BLOCK = 0.6        
    PROB_MODIFY_BLOCK = 0.8         
    
    # ==================== NTK评估配置 ====================
    NTK_BATCH_SIZE = 64              
    NTK_INPUT_SIZE = (3, 32, 32)    
    NTK_NUM_CLASSES = 10            
    NTK_PARAM_THRESHOLD = 10000000  # 提高阈值，避免太多模型被跳过
    
    # ==================== 训练参数 ====================
    DEVICE = 'cuda'                 
    BATCH_SIZE = 128             
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    ADAMW_BETAS = (0.9, 0.999)
    ADAMW_EPS = 1e-8
    
    # ==================== 早停参数 ====================
    EARLY_STOPPING_ENABLED = True   # 是否启用早停
    EARLY_STOPPING_PATIENCE = 50    # 早停耐心值（连续多少轮无提升则停止）
    EARLY_STOPPING_MIN_DELTA = 0.01 # 最小提升阈值（%），低于此值不算提升             
    
    # ==================== ImageNet 专用参数 ====================
    IMAGENET_ROOT = os.path.join(DATA_DIR, 'imagenet')  # ImageNet 数据集根目录
    IMAGENET_BATCH_SIZE =64           # ImageNet 批次大小（显存考虑）
    IMAGENET_INPUT_SIZE = 224          # ImageNet 输入尺寸
    IMAGENET_NUM_CLASSES = 1000        # ImageNet 类别数
    
    # ==================== 最终评估参数 ====================
    FINAL_DATASET = 'cifar10'       
    
    # ==================== SENet参数 ====================
    SENET_REDUCTION = 16            
    
    # ==================== 日志参数 ====================
    LOG_DIR = os.path.join(BASE_DIR, 'logs')              
    LOG_LEVEL = 'INFO'              
    SAVE_CHECKPOINT = True          
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')  
    
    # ==================== TensorBoard参数 ====================
    USE_TENSORBOARD = True          
    TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')      
    
    # ==================== 调试参数 ====================
    SAVE_FAILED_INDIVIDUALS = True  
    FAILED_INDIVIDUALS_DIR = os.path.join(BASE_DIR, 'failed_individuals')  
    
    # ==================== 架构约束参数 ====================
    MIN_FEATURE_SIZE = 1            
    INPUT_IMAGE_SIZE = 32
    MAX_CHANNELS = 1024             # 最大通道数限制，防止 concat 模式下通道爆炸           
    
    # ==================== 其他参数 ====================
    RANDOM_SEED = random.randint(0, 2**32 - 1)  
    NUM_WORKERS = 8                 

    def get_search_space_summary(self) -> str:
        """获取搜索空间的摘要字符串"""
        # Simplified for now, can be expanded
        return "Search Space Summary"

# 全局配置实例
config = Config()
