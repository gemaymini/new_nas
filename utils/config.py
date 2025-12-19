# -*- coding: utf-8 -*-
"""
神经网络架构搜索算法 - 配置文件
包含所有超参数配置
"""
import random

class Config:
    """
    配置类，包含所有超参数
    """
    
    # ==================== 进化算法参数 ====================
    POPULATION_SIZE = 50          # 种群大小
    MAX_GEN = 100               # 最大进化代数
    G1 = 30                       # 第一阶段结束代数
    G2 = 40                         # 第二阶段结束代数
    TOURNAMENT_SIZE = 5             # 锦标赛选择的个体数量
    TOURNAMENT_WINNERS = 2          # 锦标赛选择的胜者数量
    
    # ==================== 交叉算子参数 ====================
    PROB_CROSSOVER = 0.6            # 交叉概率
    PROB_MUTATION = 0.4             # 变异概率
    CROSSOVER_TYPE = 'unit'         # 交叉类型
    
    # ==================== 搜索空间参数 ====================
    MIN_UNIT_NUM = 2                # 最小unit数量
    MAX_UNIT_NUM = 5               # 最大unit数量
    
    MIN_BLOCK_NUM = 1               # 每个unit最小block数量
    MAX_BLOCK_NUM = 5               # 每个unit最大block数量
    
    CHANNEL_OPTIONS = [16, 32, 64, 128, 256, 512]
    GROUP_OPTIONS = [1, 2, 4, 8, 16, 32]
    POOL_TYPE_OPTIONS = [0, 1]
    POOL_STRIDE_OPTIONS = [1, 2]
    SENET_OPTIONS = [0, 1]
    
    # 初始卷积层参数
    INIT_CONV_OUT_CHANNELS = 32     
    INIT_CONV_KERNEL_SIZE = 3       
    INIT_CONV_STRIDE = 1            
    INIT_CONV_PADDING = 1           
    
    # ==================== 变异概率参数 ====================
    PROB_SWAP_BLOCKS = 0.4          
    PROB_SWAP_UNITS = 0.4          
    PROB_ADD_UNIT = 0.2             
    PROB_ADD_BLOCK = 0.4           
    PROB_DELETE_UNIT = 0.2          
    PROB_DELETE_BLOCK = 0.4        
    PROB_MODIFY_BLOCK = 0.5         
    
    # ==================== 自适应变异参数 ====================
    ADAPTIVE_MUTATION = True        
    MUTATION_SCALE_PHASE1 = 1.0     
    MUTATION_SCALE_PHASE2 = 0.8     
    MUTATION_SCALE_PHASE3 = 0.6     
    STAGNATION_THRESHOLD = 5        
    STAGNATION_MUTATION_BOOST = 1.5 
    
    # ==================== 内存优化参数 ====================
    MAX_NETWORK_SIZE_MB = 50        
    NTK_BATCH_SIZE = 4              
    GPU_MEMORY_CHECK_INTERVAL = 10  
    FORCE_CPU_EVAL_THRESHOLD = 100  
    NTK_INPUT_SIZE = (3, 32, 32)    
    NTK_NUM_CLASSES = 10            
    NTK_PARAM_THRESHOLD = 10000000  
    
    # ==================== 阶段2快速评估参数 ====================
    PHASE2_QUICK_EVAL_SAMPLES = 1024   
    PHASE2_QUICK_EVAL_EPOCHS = 3      
    PHASE2_QUICK_EVAL_BATCH_SIZE = 32 
    
    # ==================== 训练参数 ====================
    DEVICE = 'cuda'                 
    BATCH_SIZE = 128                
    LEARNING_RATE = 0.1             
    MOMENTUM = 0.9                  
    WEIGHT_DECAY = 5e-4             
    
    # ==================== 最终评估参数 ====================
    FINAL_TOP_K = 1                 
    FINAL_TRAIN_EPOCHS = 200         
    FINAL_DATASET = 'cifar10'       
    
    # ==================== SENet参数 ====================
    SENET_REDUCTION = 16            
    
    # ==================== 日志参数 ====================
    LOG_DIR = './logs'              
    LOG_LEVEL = 'INFO'              
    SAVE_CHECKPOINT = True          
    CHECKPOINT_DIR = './checkpoints'  
    
    # ==================== TensorBoard参数 ====================
    USE_TENSORBOARD = True          
    TENSORBOARD_DIR = './runs'      
    
    # ==================== 调试参数 ====================
    SAVE_FAILED_INDIVIDUALS = True  
    FAILED_INDIVIDUALS_DIR = './failed_individuals'  
    
    # ==================== 架构约束参数 ====================
    MIN_FEATURE_SIZE = 1            
    INPUT_IMAGE_SIZE = 32           
    
    # ==================== 其他参数 ====================
    RANDOM_SEED = random.randint(0, 2**32 - 1)  
    NUM_WORKERS = 4                 

    def get_search_space_summary(self) -> str:
        """获取搜索空间的摘要字符串"""
        # Simplified for now, can be expanded
        return "Search Space Summary"

# 全局配置实例
config = Config()
