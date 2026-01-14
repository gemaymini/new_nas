  1. 降低 AdamW 基础学习率，配合 warmup+cosine；调 weight decay（如 1e-2→5e-3/1e-3）。
  2. 对齐训练设置：相同的 LR schedule、batch、数据增强、正则，再做公平对比。
  3. 若 batch 大，调小 LR 或用梯度裁剪/噪声；若过拟合，提升增强或加正则。
  4. 观察训练/验证曲线：若训练 acc 高、验证 acc 低→过拟合；若收敛慢→LR 过小或 schedule 不佳。 
  5. 请完整修改代码，将训练模型使用的adamw优化器改回sgd优化器