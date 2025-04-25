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
    prompt_field: str = "prompt",
    content_field: str = "content",
):
    """
    加载SFT数据集
    
    Args:
        dataset_name: 数据集名称或路径
        tokenizer: 分词器，用于处理文本
        max_seq_length: 最大序列长度
        format: 数据格式，可以是"instruction"、"messages"、"text"或"prompt_content"
        dataset_config_name: 数据集配置名称
        split: 数据集分割
        streaming: 是否使用流式加载
        max_samples: 最大样本数量
        preprocessing_num_workers: 预处理工作线程数
        prompt_field: 提示字段名称
        content_field: 内容字段名称
        
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
    
    # 如果是prompt_content格式，添加数据处理逻辑
    if format == "prompt_content" and tokenizer is not None:
        logger.info(f"Processing prompt-content format with fields: {prompt_field} and {content_field}")
        
        def process_prompt_content(examples):
            prompts = examples[prompt_field]
            contents = examples[content_field]
            
        # 组合提示和内容
        combined_texts = []
        for prompt, content in zip(prompts, contents):
            # 为Qwen模型格式化提示和内容 - 使用Qwen2.5的chat模板
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>"
            combined_texts.append(text) 
            
            # 分词
            tokenized = tokenizer(
                combined_texts,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            # 创建标签（用于因果语言模型训练）
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 应用处理函数
        if tokenizer:
            train_dataset = train_dataset.map(
                process_prompt_content,
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Processing dataset",
                num_proc=preprocessing_num_workers,
            )
            
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    process_prompt_content,
                    batched=True,
                    remove_columns=eval_dataset.column_names,
                    desc="Processing evaluation dataset",
                    num_proc=preprocessing_num_workers,
                )
    
    return train_dataset, eval_dataset
