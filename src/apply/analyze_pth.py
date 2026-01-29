"""
分析 .pth 文件结构的脚本
用于探索未知 PyTorch 模型文件的内容和结构
"""
import torch
import sys


def analyze_pth(file_path):
    """分析 .pth 文件的结构和内容"""
    print(f"分析文件: {file_path}")
    print("=" * 60)
    
    # 加载文件
    data = torch.load(file_path, map_location='cpu')
    
    # 查看顶层类型
    print(f"\n顶层类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"\n顶层键: {list(data.keys())}")
        print("-" * 60)
        
        for key, value in data.items():
            print(f"\n键: '{key}'")
            print(f"  类型: {type(value)}")
            
            if isinstance(value, torch.Tensor):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
            elif isinstance(value, dict):
                print(f"  子键数量: {len(value)}")
                if len(value) <= 10:
                    for sub_key in value.keys():
                        sub_val = value[sub_key]
                        if isinstance(sub_val, torch.Tensor):
                            print(f"    {sub_key}: {sub_val.shape}")
                        else:
                            print(f"    {sub_key}: {type(sub_val)}")
                else:
                    # 打印前10个键
                    print("  前10个子键:")
                    for i, sub_key in enumerate(list(value.keys())[:10]):
                        sub_val = value[sub_key]
                        if isinstance(sub_val, torch.Tensor):
                            print(f"    {sub_key}: {sub_val.shape}")
                        else:
                            print(f"    {sub_key}: {type(sub_val)}")
                    print(f"  ... 还有 {len(value) - 10} 个键")
            elif isinstance(value, (list, tuple)):
                print(f"  长度: {len(value)}")
                if len(value) > 0:
                    print(f"  第一个元素类型: {type(value[0])}")
                    if len(value) <= 5:
                        for i, item in enumerate(value):
                            print(f"    [{i}]: {item}")
            elif isinstance(value, str):
                print(f"  值: '{value}'")
            elif isinstance(value, (int, float)):
                print(f"  值: {value}")
            else:
                print(f"  值: {value}")
        
        # 如果有 state_dict 或 model_state_dict，详细分析网络结构
        state_dict_key = None
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in data and isinstance(data[key], dict):
                state_dict_key = key
                break
        
        if state_dict_key:
            print("\n" + "=" * 60)
            print(f"分析网络结构 (从 '{state_dict_key}'):")
            print("=" * 60)
            analyze_state_dict(data[state_dict_key])
        elif all(isinstance(v, torch.Tensor) for v in data.values()):
            # 整个 data 就是 state_dict
            print("\n" + "=" * 60)
            print("分析网络结构 (整个文件是 state_dict):")
            print("=" * 60)
            analyze_state_dict(data)
    
    elif isinstance(data, torch.nn.Module):
        print("\n文件包含完整的 PyTorch 模型!")
        print(f"模型类型: {type(data)}")
        print("\n模型结构:")
        print(data)
    
    else:
        print(f"\n未知格式: {type(data)}")
        print(f"值: {data}")


def analyze_state_dict(state_dict):
    """分析 state_dict 并推断网络结构"""
    print(f"\n总共 {len(state_dict)} 个参数张量")
    print("-" * 60)
    
    # 分组显示层
    layers = {}
    for name, param in state_dict.items():
        # 提取层名称（去掉最后的 .weight 或 .bias）
        parts = name.rsplit('.', 1)
        layer_name = parts[0] if len(parts) > 1 else name
        
        if layer_name not in layers:
            layers[layer_name] = []
        layers[layer_name].append((name, param.shape))
    
    print(f"\n发现 {len(layers)} 个层:")
    for layer_name, params in layers.items():
        print(f"\n  {layer_name}:")
        for param_name, shape in params:
            print(f"    {param_name}: {list(shape)}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_pth.py <pth文件路径>")
        print("示例: python analyze_pth.py model.pth")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_pth(file_path)
