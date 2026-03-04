import json
import random
import os

def merge_alpaca_datasets(file_paths, output_file, sample_size=2500, seed=42):
    """
    从多个 Alpaca 格式的 JSON 数据集中各抽取指定数量的数据并合并。
    
    Args:
        file_paths (list): 输入文件的路径列表。
        output_file (str): 输出文件的路径。
        sample_size (int): 每个文件抽取的样本数量（默认 2500）。
        seed (int): 随机种子，保证结果可复现。
    """
    # 设置随机种子以保证结果可复现
    random.seed(seed)
    
    merged_data = []
    total_extracted = 0
    
    print(f"🚀 开始处理，目标是从每个文件抽取 {sample_size} 条数据...")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️  警告: 文件未找到，跳过: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查数据是否为列表格式
            if not isinstance(data, list):
                print(f"⚠️  警告: 文件格式错误（应为 JSON 列表），跳过: {file_path}")
                continue
                
            original_len = len(data)
            
            # 确定抽取数量：如果数据不足 2500 条，则全部抽取
            current_sample_size = min(original_len, sample_size)
            
            if original_len < sample_size:
                print(f"ℹ️  提示: {file_path} 数据量不足 {sample_size} 条 (只有 {original_len} 条)，将全部使用。")
            
            # 随机抽样
            sampled_data = random.sample(data, current_sample_size)
            merged_data.extend(sampled_data)
            
            print(f"✅ 已从 {os.path.basename(file_path)} 抽取 {current_sample_size} 条数据")
            total_extracted += current_sample_size
            
        except json.JSONDecodeError:
            print(f"❌ 错误: {file_path} 不是有效的 JSON 文件。")
        except Exception as e:
            print(f"❌ 处理 {file_path} 时发生未知错误: {e}")

    # 再次打乱最终的合并数据，避免相同来源的数据聚集在一起
    random.shuffle(merged_data)
    
    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(merged_data, f_out, ensure_ascii=False, indent=4)
        print("-" * 30)
        print(f"🎉 成功！新数据集已保存至: {output_file}")
        print(f"📊 总数据量: {len(merged_data)} 条")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

# ================= 配置区域 =================

# 请在这里替换为你实际的文件路径
input_files = [
    "/data/mxy/workspace/L-MTP-main/LLaMA-Factory/data/alpaca_openpangu_deepctrl_clean.json",
    "/data/mxy/workspace/L-MTP-main/LLaMA-Factory/data/alpaca_openpangu_Infinity-Instruct-7M_clean.json",
    "/data/mxy/workspace/L-MTP-main/LLaMA-Factory/data/alpaca_openpangu_moss_clean.json",
    "/data/mxy/workspace/L-MTP-main/LLaMA-Factory/data/alpaca_openpangu_ultrachat_clean.json"
]

output_filename = "/data/mxy/workspace/L-MTP-main/LLaMA-Factory/data_alpaca_merged_10k.json"

# ================= 运行脚本 =================

if __name__ == "__main__":
    merge_alpaca_datasets(input_files, output_filename, sample_size=2500)