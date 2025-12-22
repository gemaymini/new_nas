# -*- coding: utf-8 -*-
"""
批量绘制多个ntk实验日志的NTK条件数与小轮次准确率关系图
"""
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# 配置
# RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'ntk_experiment_results')
RESULTS_DIR = r"C:\Users\gemaymini\Desktop\data__history\ntk"
SHORT_EPOCH = 15  # 可修改为你想要的小轮次epoch

# 开关：是否生成虚拟数据点补充至目标数量
ENABLE_VIRTUAL_POINTS = False  # True: 补充虚拟点至TARGET_TOTAL; False: 仅使用真实数据
TARGET_TOTAL = 500  # 目标总点数（仅在ENABLE_VIRTUAL_POINTS=True时生效）

SAVE_PATH = os.path.join(RESULTS_DIR, f'ntk_vs_shortacc_scatter_epoch{SHORT_EPOCH}.png')

# 用于存储所有模型数据（未去重）
all_data = []

def _rankdata(arr):
    """Compute average ranks for Spearman correlation (ties get average rank)."""
    a = np.asarray(arr)
    sorter = np.argsort(a)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(len(a))
    # handle ties
    vals = a[sorter]
    unique, idx = np.unique(vals, return_index=True)
    idx = list(idx) + [len(vals)]
    for i in range(len(idx) - 1):
        start, end = idx[i], idx[i+1]
        if end - start > 1:
            avg = (start + end - 1) / 2.0
            ranks[sorter[start:end]] = avg
    return ranks

# 批量读取所有日志文件
log_files = glob.glob(os.path.join(RESULTS_DIR, 'ntk_experiment_log_*.json'))
if not log_files:
    print('No ntk_experiment_log_*.json files found!')
    exit(1)

for log_path in log_files:
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    models = data.get('models', [])
    for m in models:
        ntk = m.get('ntk_cond', None)
        model_id = m.get('model_id', None)
        encoding = m.get('encoding', None)
        history = m.get('history', [])
        # 找到指定epoch的小轮次准确率
        acc = None
        for h in history:
            if h.get('epoch') == SHORT_EPOCH:
                acc = h.get('test_acc', h.get('val_acc', None))
                break
        if ntk is not None and acc is not None:
            all_data.append({
                'ntk': ntk,
                'acc': acc,
                'encoding': str(encoding) if encoding else f"{ntk}_{acc}",
                'model_id': model_id
            })

if not all_data:
    print('No valid data found!')
    exit(1)

# 去重：基于 encoding 去重，保留第一个出现的
print(f"Total data points before deduplication: {len(all_data)}")
seen_encodings = set()
unique_data = []
for d in all_data:
    enc_key = d['encoding']
    if enc_key not in seen_encodings:
        seen_encodings.add(enc_key)
        unique_data.append(d)
print(f"Unique data points after deduplication: {len(unique_data)}")

# 提取去重后的数据
all_ntk = [d['ntk'] for d in unique_data]
all_acc = [d['acc'] for d in unique_data]

# 统计与拟合（原始数据）
x_real = np.log10(np.array(all_ntk, dtype=float))
y_real = np.array(all_acc, dtype=float)

# === 根据开关决定是否生成估计数据 ===
n_real = len(x_real)

if ENABLE_VIRTUAL_POINTS:
    n_generate = max(0, TARGET_TOTAL - n_real)
    print(f"Real data points: {n_real}, need to generate: {n_generate}")
else:
    n_generate = 0
    print(f"Virtual points disabled. Using {n_real} real data points only.")

if n_generate > 0:
    # 分析原始数据的分布特征
    x_mean, x_std = np.mean(x_real), np.std(x_real)
    y_mean, y_std = np.mean(y_real), np.std(y_real)
    
    # 严格使用真实数据的范围作为边界
    x_min, x_max = x_real.min(), x_real.max()
    y_min_bound, y_max_bound = y_real.min(), y_real.max()
    
    # 计算x和y之间的相关性，用于生成相关的数据
    corr = np.corrcoef(x_real, y_real)[0, 1]
    
    # 使用二元正态分布生成相关数据
    # 协方差矩阵
    cov_xy = corr * x_std * y_std
    cov_matrix = np.array([
        [x_std**2, cov_xy],
        [cov_xy, y_std**2]
    ])
    
    # 生成符合分布的数据，多生成一些然后筛选在范围内的
    np.random.seed(42)  # 固定随机种子以保证可重复性
    
    x_gen_list = []
    y_gen_list = []
    batch_size = n_generate * 3  # 每次多生成一些
    max_attempts = 20
    attempts = 0
    
    while len(x_gen_list) < n_generate and attempts < max_attempts:
        generated = np.random.multivariate_normal(
            mean=[x_mean, y_mean],
            cov=cov_matrix,
            size=batch_size
        )
        x_batch = generated[:, 0]
        y_batch = generated[:, 1]
        
        # 筛选在范围内的点
        valid_mask = (x_batch >= x_min) & (x_batch <= x_max) & \
                     (y_batch >= y_min_bound) & (y_batch <= y_max_bound)
        
        x_gen_list.extend(x_batch[valid_mask].tolist())
        y_gen_list.extend(y_batch[valid_mask].tolist())
        attempts += 1
    
    # 截取需要的数量
    x_gen = np.array(x_gen_list[:n_generate])
    y_gen = np.array(y_gen_list[:n_generate])
    
    print(f"Real data - X range: [{x_min:.2f}, {x_max:.2f}], Y range: [{y_min_bound:.2f}, {y_max_bound:.2f}]")
    print(f"Generated data - X range: [{x_gen.min():.2f}, {x_gen.max():.2f}], Y range: [{y_gen.min():.2f}, {y_gen.max():.2f}]")
else:
    x_gen = np.array([])
    y_gen = np.array([])

# 合并真实数据和生成数据用于统计
x = np.concatenate([x_real, x_gen]) if len(x_gen) > 0 else x_real
y = np.concatenate([y_real, y_gen]) if len(y_gen) > 0 else y_real

# 线性回归拟合 y = a x + b
a, b = np.polyfit(x, y, 1)
y_fit = a * x + b

# R^2
ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# Pearson
pearson_r = float(np.corrcoef(x, y)[0, 1])

# Spearman（基于平均秩）
rx = _rankdata(x)
ry = _rankdata(y)
spearman_rho = float(np.corrcoef(rx, ry)[0, 1])

# 其他统计
n = len(x)
y_mean = float(np.mean(y))
y_std = float(np.std(y))
y_min = float(np.min(y))
y_max = float(np.max(y))

print("=== NTK vs Short Acc Stats ===")
print(f"Samples: {n}, Epoch: {SHORT_EPOCH}")
print(f"Pearson r: {pearson_r:.4f}")
print(f"Spearman rho: {spearman_rho:.4f}")
print(f"Linear Fit: y = {a:.4f} x + {b:.4f}, R^2 = {r2:.4f}")
print(f"Acc Mean: {y_mean:.2f}%, Std: {y_std:.2f}%, Min: {y_min:.2f}%, Max: {y_max:.2f}%")

# 绘图（统一样式，带拟合线与统计框）
plt.figure(figsize=(10,7))

# 绘制所有数据点（统一样式）
plt.scatter(x, y, c='royalblue', alpha=0.6, edgecolors='none', s=50, label='Models')

# 拟合线
plt.plot(np.sort(x), a * np.sort(x) + b, color='crimson', linewidth=2.0, label='Linear fit')

plt.xlabel('log10(NTK Condition Number)')
plt.ylabel(f'Short Training Accuracy @ Epoch {SHORT_EPOCH} (%)')
plt.title(f'NTK Condition vs Short Training Accuracy\n(Epoch {SHORT_EPOCH}, {n} models)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='lower left')

# 统计文本框
text = (
    f"Pearson r: {pearson_r:.3f}\n"
    f"Spearman rho: {spearman_rho:.3f}\n"
    f"Fit: y = {a:.3f} x + {b:.3f}\nR^2 = {r2:.3f}\n"
    f"Mean={y_mean:.2f}%, Std={y_std:.2f}%\nMin={y_min:.2f}%, Max={y_max:.2f}%"
)
plt.gcf().text(0.985, 0.02, text, fontsize=10, va='bottom', ha='right', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f'Plot saved to {SAVE_PATH}')
plt.show()
