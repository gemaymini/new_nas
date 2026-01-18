# -*- coding: utf-8 -*-
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoding import Individual
from engine.evaluator import NTKEvaluator
from configuration.config import config

# ==========================================
# 在这里修改你的模型编码列表
# ==========================================
TARGET_ENCODING = [1, 1, 64, 1, 0, 1, 0, 0, 0, 0, 3, 1] 
# ==========================================

def main():
    # 打印当前设置
    print(f"Target Encoding: {TARGET_ENCODING}")
    print(f"Dataset: {config.FINAL_DATASET}")
    print(f"Device: {config.DEVICE}")

    try:
        # 创建个体对象
        individual = Individual(TARGET_ENCODING)
        individual.id = 999  # Dummy ID

        # 初始化评估器
        # 注意：evaluator 内部会使用修改后的 compute_ntk_score (12次计算, 去极值, 取平均)
        evaluator = NTKEvaluator(dataset=config.FINAL_DATASET, device=config.DEVICE)
        
        print("\nStarting NTK calculation...")
        print("Note: Computing 12 runs, removing min/max, averaging remaining 10...")
        
        # 计算分数
        score = evaluator.evaluate_individual(individual)
        
        print("-" * 30)
        print(f"Final NTK Score (Log10 Cond): {score}")
        print("-" * 30)

    except Exception as e:
        print(f"Error during NTK calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()