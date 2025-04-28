#!/usr/bin/env python
# 创建自定义数据集配置，用于处理Hugging Face上的数据集

from datasets import load_dataset

def load_custom_dataset(dataset_name, max_samples=None):
    """
    加载自定义数据集并转换为LLaMA Factory所需的格式
    
    Args:
        dataset_name: Hugging Face上的数据集名称
        max_samples: 最大样本数，用于测试
    
    Returns:
        处理后的数据集
    """
    # 加载数据集
    dataset = load_dataset(dataset_name)
    
    # 转换为LLaMA Factory所需的格式
    def convert_format(example):
        return {
            "instruction": example["prompt"],
            "input": "",  # 可选，如果数据集中有input列，可以使用
            "output": example["content"]
        }
    
    # 转换训练集
    train_dataset = dataset["train"].map(convert_format)
    
    # 如果指定了最大样本数，则只使用前max_samples个样本
    if max_samples is not None and max_samples > 0:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
    
    return train_dataset

if __name__ == "__main__":
    # 测试代码
    dataset = load_custom_dataset("ChizhongWang/secondKarlMarx-sft", max_samples=2)
    print(f"加载了 {len(dataset)} 个样本")
    print("样本示例:")
    print(dataset[0])
