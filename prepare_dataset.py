#!/usr/bin/env python
# 从Hugging Face下载数据集并转换为LLaMA Factory支持的格式

import os
import json
from datasets import load_dataset

def download_and_convert_dataset(dataset_name, output_dir="./training", max_samples=None):
    """
    下载Hugging Face数据集并转换为LLaMA Factory支持的格式
    
    Args:
        dataset_name: Hugging Face上的数据集名称
        output_dir: 输出目录
        max_samples: 最大样本数，用于测试
    """
    print(f"下载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为LLaMA Factory支持的格式
    converted_data = []
    
    # 获取训练集
    train_dataset = dataset["train"]
    
    # 如果指定了最大样本数，则只使用前max_samples个样本
    if max_samples is not None and max_samples > 0:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
    
    print(f"处理 {len(train_dataset)} 个样本")
    
    for example in train_dataset:
        converted_example = {
            "instruction": example["prompt"],
            "input": "",  # 可选，如果数据集中有input列，可以使用
            "output": example["content"]
        }
        converted_data.append(converted_example)
    
    # 保存为JSON文件
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    # 创建dataset_info.json文件
    dataset_info = {
        "alpaca": {
            "file_name": "train.json",
            "file_sha1": None
        }
    }
    
    info_file = os.path.join(output_dir, "dataset_info.json")
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存到: {output_file}")
    print(f"数据集信息已保存到: {info_file}")
    print(f"样本数量: {len(converted_data)}")
    
    # 打印第一个样本作为示例
    if converted_data:
        print("\n示例:")
        print(json.dumps(converted_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 下载并转换数据集
    download_and_convert_dataset("ChizhongWang/secondKarlMarx-sft", max_samples=None)
