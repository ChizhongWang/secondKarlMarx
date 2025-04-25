"""
数据处理工具 - 用于加载和预处理训练数据
"""

import logging
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def load_sft_dataset(
    dataset_name: str,
    tokenizer=None,
    max_seq_length: int = 4096,
    format: str = "instruction",
    dataset_config_name: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    preprocessing_num_workers: Optional[int] = None,
):
    """
    加载SFT数据集
    
    Args:
        dataset_name: 数据集名称或路径
        tokenizer: 分词器，用于处理文本
        max_seq_length: 最大序列长度
        format: 数据格式，可以是"instruction"、"messages"或"text"
        dataset_config_name: 数据集配置名称
        split: 数据集分割
        streaming: 是否使用流式加载
        max_samples: 最大样本数量
        preprocessing_num_workers: 预处理工作线程数
        
    Returns:
        train_dataset: 训练数据集
        eval_dataset: 评估数据集（如果有）
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # 加载数据集
    if streaming:
        ds = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            streaming=True,
            use_auth_token=True,
        )
        # 流式数据集不支持分割，因此我们手动分割
        ds = ds.shuffle(seed=42, buffer_size=10000)
        train_dataset = ds
        eval_dataset = None
    else:
        ds = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            use_auth_token=True,
        )
        
        # 如果提供了最大样本数，则截取数据集
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        # 分割数据集为训练集和验证集
        if "train" in ds:
            train_dataset = ds["train"]
            eval_dataset = ds.get("validation", None)
        else:
            # 如果没有预定义的分割，则手动分割
            ds = ds.shuffle(seed=42)
            split_dataset = ds.train_test_split(test_size=0.05)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
    
    logger.info(f"Dataset loaded with {len(train_dataset) if not streaming else 'streaming'} training examples")
    if eval_dataset:
        logger.info(f"Evaluation dataset has {len(eval_dataset)} examples")
    
    # 处理长文本
    logger.info(f"Max sequence length is set to {max_seq_length}")
    logger.info(f"Data format: {format}")
    
    # 如果需要，可以在这里添加数据预处理逻辑
    # 例如，将数据转换为SFT所需的格式
    
    return train_dataset, eval_dataset
