"""
从 .pth 文件中提取 DARTS Cell 编码信息
"""
import torch
import json
import sys


def extract_encoding(file_path):
    """从 .pth 文件中提取编码信息"""
    print(f"从文件提取编码: {file_path}")
    print("=" * 60)
    
    # 加载文件
    data = torch.load(file_path, map_location='cpu')
    
    if not isinstance(data, dict):
        print("错误: 文件不是字典格式")
        return None
    
    result = {}
    
    # 提取 genotype
    if 'genotype' in data:
        print("\n[Genotype]")
        genotype = data['genotype']
        result['genotype'] = genotype
        if isinstance(genotype, dict):
            for key, value in genotype.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {genotype}")
    
    # 提取 normal_cell
    if 'normal_cell' in data:
        print("\n[Normal Cell Encoding]")
        normal_cell = data['normal_cell']
        result['normal_cell'] = normal_cell
        print(f"  类型: {type(normal_cell)}")
        print(f"  内容: {normal_cell}")
    
    # 提取 reduction_cell
    if 'reduction_cell' in data:
        print("\n[Reduction Cell Encoding]")
        reduction_cell = data['reduction_cell']
        result['reduction_cell'] = reduction_cell
        print(f"  类型: {type(reduction_cell)}")
        print(f"  内容: {reduction_cell}")
    
    # 提取准确率
    if 'accuracy' in data:
        print(f"\n[Accuracy]: {data['accuracy']}")
        result['accuracy'] = data['accuracy']
    
    # 提取参数量
    if 'param_count' in data:
        print(f"[Param Count]: {data['param_count']:,}")
        result['param_count'] = data['param_count']
    
    # 提取训练历史
    if 'history' in data:
        history = data['history']
        result['history'] = history
        print(f"\n[Training History]")
        if isinstance(history, dict):
            for key, value in history.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} epochs")
                    if len(value) > 0:
                        print(f"    最后5个: {value[-5:]}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(history, list):
            print(f"  {len(history)} epochs")
    
    # 保存编码到 JSON 文件
    output_path = file_path.replace('.pth', '_encoding.json')
    
    # 转换为可序列化格式
    serializable_result = {}
    for key, value in result.items():
        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_result[key] = value
        else:
            serializable_result[key] = str(value)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n编码信息已保存到: {output_path}")
    
    return result


def format_cell_encoding(cell_encoding):
    """格式化 Cell 编码为可读格式"""
    if not isinstance(cell_encoding, list):
        return str(cell_encoding)
    
    lines = []
    for i, edge in enumerate(cell_encoding):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            op_name, from_node = edge[0], edge[1]
            lines.append(f"  Edge {i}: {op_name} ← Node {from_node}")
        else:
            lines.append(f"  Edge {i}: {edge}")
    return '\n'.join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_encoding.py <pth文件路径>")
        print("示例: python extract_encoding.py model.pth")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = extract_encoding(file_path)
    
    if result:
        print("\n" + "=" * 60)
        print("格式化的 Cell 编码:")
        print("=" * 60)
        
        if 'normal_cell' in result:
            print("\n[Normal Cell]")
            print(format_cell_encoding(result['normal_cell']))
        
        if 'reduction_cell' in result:
            print("\n[Reduction Cell]")
            print(format_cell_encoding(result['reduction_cell']))
